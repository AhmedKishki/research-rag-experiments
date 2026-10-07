"""Run the settings resolution inside a named tree, and bring the answer back.

The parent process holds the installed app, not the tree an arm measures, so it
cannot resolve that tree's settings itself. This spawns the resolution module
with the tree first on `PYTHONPATH` and reads the answer back as JSON. The
interpreter is this toolkit's own, because that is the environment holding the
app's dependencies, and the tree only has to precede it on the path.

The environment a child is given is stated rather than inherited silently. Two
prefixes are dropped, two variables are set, and the record says which: a run whose
child environment nobody can describe is a run nobody can reproduce.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import resource_limits
from ..errors import ExperimentError
from ..niceness import run_low_priority
from .locate import ENGINE_HARNESS_ENV, EngineSource

#: The module run inside the tree. Named here once so the parent and the child
#: cannot disagree about it.
RESOLVER_MODULE = "rag_experiments.engine.resolve_settings"

#: The prefixes the app reads its settings and its engine's roots from. The first
#: is the app's own settings layer; the second is what the app's harness reads
#: `--model-cache-root`, `--runtime-root`, and `--dense-backend` from, so an
#: exported variable would relocate the cache or the runtime a measurement reads
#: without any of it appearing in the record.
SETTINGS_PREFIXES = ("RESEARCH_RAG_", "RESEARCH_ULTRARAG_")

#: The prefix this module drops, named for the record.
SETTINGS_PREFIX = SETTINGS_PREFIXES[0]

#: The variable that relocates the account configuration directory. The account
#: overlay is resolved beneath it, so pointing it at an empty directory is what
#: keeps one account's settings out of a measured run. It is not applied to the
#: resolution itself: a baseline is what a reader gets today, account settings
#: included, and it is pinned into the sandbox from there.
CONFIG_HOME_ENV = "XDG_CONFIG_HOME"

#: The variable that keeps an import from writing bytecode beside the tree it
#: read. The engine under test is often a developer's working checkout, and a
#: `__pycache__` written into it is a write into the code a run measured.
BYTECODE_ENV = "PYTHONDONTWRITEBYTECODE"

#: How long a resolution is given. It reads a TOML file and coerces forty values,
#: so a longer wait means the child is not this module.
RESOLVE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True, slots=True)
class SettingsAnswer:
    """What the tree resolved, and how it says each key reached its value."""

    document: str
    values: dict[str, Any]
    baseline_values: dict[str, Any]
    baseline_layers: dict[str, str]
    layers: dict[str, str]
    overridden: tuple[str, ...]
    model_cache_root: str | None

    def describe(self) -> dict[str, Any]:
        """The record form, with the document's own digest rather than its text."""

        import hashlib

        return {
            "key_count": len(self.values),
            "document_sha256": hashlib.sha256(self.document.encode()).hexdigest(),
            "overridden": list(self.overridden),
            "baseline_layers": self.baseline_layers,
            "arm_layers": self.layers,
            "model_cache_root": self.model_cache_root,
            "values": self.values,
            "baseline_values": self.baseline_values,
        }


def resolve_settings_for_tree(
    engine: EngineSource,
    *,
    project: Path,
    overlay: dict[str, Any],
    pin_project: Path | None = None,
) -> SettingsAnswer:
    """Resolve `project`'s settings as `engine` would, then apply `overlay`.

    The project handed in is the **source** project, not the sandbox: the baseline
    is the behaviour a reader gets today, from the packaged defaults, the account
    overlay, and the project's own file. Nothing is created on the way, and the
    environment layer is dropped, so no exported variable can move the numbers.

    `pin_project` is the sandbox the values will be pinned into. The model cache is
    resolved against it, so the pinned file carries the absolute path of the
    shared cache rather than an empty or relative one a relocated configuration
    home could strand.
    """

    request = json.dumps(
        {
            "project": str(project),
            "overlay": overlay,
            "pin_project": None if pin_project is None else str(pin_project),
        },
        ensure_ascii=False,
    )
    environment = child_environment(engine)
    try:
        completed = run_low_priority(
            [sys.executable, "-m", RESOLVER_MODULE],
            input=request,
            capture_output=True,
            text=True,
            env=environment,
            timeout=RESOLVE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExperimentError(
            f"Resolving settings in {engine.root} took more than "
            f"{RESOLVE_TIMEOUT_SECONDS} s: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise ExperimentError(
            f"{engine.root} refused the settings an arm asked for:\n"
            f"{_last_lines(completed.stderr)}"
        )
    try:
        answer = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExperimentError(
            f"The settings answer from {engine.root} was not JSON: {exc}"
        ) from exc
    if not isinstance(answer, dict) or "document" not in answer:
        raise ExperimentError(
            f"The settings answer from {engine.root} carried no document, so there "
            "is nothing to pin."
        )
    return SettingsAnswer(
        document=str(answer["document"]),
        values=dict(answer["values"]),
        baseline_values=dict(answer.get("baseline_values") or {}),
        baseline_layers=dict(answer["baseline_layers"]),
        layers=dict(answer.get("arm_layers") or {}),
        overridden=tuple(str(key) for key in answer["overridden"]),
        model_cache_root=(
            None
            if answer.get("model_cache_root") is None
            else str(answer["model_cache_root"])
        ),
    )


def child_environment(
    engine: EngineSource,
    *,
    config_home: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """The environment a child measuring through `engine` runs with.

    Five things are deliberate here, and each is recorded by
    `describe_child_environment` so a record can say what a child was given.

    - The tree's `src` is prepended to `PYTHONPATH`, so the app and the gateway it
      spawns both import the tree rather than the installed distribution.
    - Every `RESEARCH_RAG_` and `RESEARCH_ULTRARAG_` variable is dropped. The first
      is the settings layer, which outranks the pinned file; the second is what
      the app's own harness reads its cache root, runtime root, and dense backend
      from. Either would decide a measurement without appearing in the record.
    - `XDG_CONFIG_HOME` points at a directory with no file in it, so the account
      overlay resolves somewhere empty. The model cache is deliberately left
      alone: those are immutable binaries, a run downloads nothing, and the
      absolute path it resolves to is pinned into the sandbox's settings.
    - `PYTHONDONTWRITEBYTECODE` is set, because the tree under test is frequently a
      developer's working checkout and an import must not write beside it.
    - The default thread count of the common numerical libraries is capped, so a
      library that honours `OMP_NUM_THREADS` and its peers does not use every core
      for one measured child. A library that ignores them is still bounded by the
      cgroup CPU quota the child tree runs under; the cap here is the cheap half.
      A variable the reader already set is left alone.
    - Nothing else is touched, so a child inherits the machine's own paths and
      credentials for whatever the app legitimately needs.
    """

    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith(SETTINGS_PREFIXES):
            del environment[name]
    existing = environment.get(ENGINE_HARNESS_ENV, "")
    tree_source = str(engine.source_relative)
    environment[ENGINE_HARNESS_ENV] = (
        tree_source if not existing else f"{tree_source}{os.pathsep}{existing}"
    )
    environment[BYTECODE_ENV] = "1"
    for name, value in resource_limits.thread_environment().items():
        environment.setdefault(name, value)
    if config_home is not None:
        environment[CONFIG_HOME_ENV] = str(config_home)
    if extra:
        environment.update(extra)
    return environment


def describe_child_environment(
    engine: EngineSource, *, config_home: Path | None = None
) -> dict[str, Any]:
    """What `child_environment` changes, as names rather than as values.

    Only the names dropped and the values this module sets are reported. A full
    environment would carry whatever the reader's shell held, which is a secret
    store this record has no business copying.
    """

    dropped = sorted(
        name
        for name in os.environ
        if any(name.startswith(prefix) for prefix in SETTINGS_PREFIXES)
    )
    given = {
        ENGINE_HARNESS_ENV: str(engine.source_relative),
        BYTECODE_ENV: "1",
    }
    for name, value in resource_limits.thread_environment().items():
        if os.environ.get(name) != value:
            given[name] = value
    if config_home is not None:
        given[CONFIG_HOME_ENV] = str(config_home)
    return {
        "dropped_variables": dropped,
        "dropped_prefixes": list(SETTINGS_PREFIXES),
        "given": given,
        "resource_limits": resource_limits.describe(),
        "note": (
            "Every other variable is inherited. The model cache is not relocated: "
            "its resolved path is pinned into the sandbox's settings file."
        ),
    }


def _last_lines(text: str, count: int = 6) -> str:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(lines[-count:]) if lines else "(no message)"


__all__ = [
    "BYTECODE_ENV",
    "CONFIG_HOME_ENV",
    "RESOLVER_MODULE",
    "SETTINGS_PREFIX",
    "SETTINGS_PREFIXES",
    "SettingsAnswer",
    "child_environment",
    "describe_child_environment",
    "resolve_settings_for_tree",
]
