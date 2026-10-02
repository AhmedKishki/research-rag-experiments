"""Run the settings resolution inside a named tree, and bring the answer back.

The parent process holds the installed app, not the tree an arm measures, so it
cannot resolve that tree's settings itself. This spawns the resolution module
with the tree first on `PYTHONPATH` and reads the answer back as JSON. The
interpreter is this toolkit's own, because that is the environment holding the
app's dependencies, and the tree only has to precede it on the path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ExperimentError
from .locate import ENGINE_HARNESS_ENV, EngineSource

#: The module run inside the tree. Named here once so the parent and the child
#: cannot disagree about it.
RESOLVER_MODULE = "rag_experiments.engine.resolve_settings"

#: The prefix the app's environment layer reads. Declared here because the app
#: names it as one of the names it keeps, and a run that dropped the wrong prefix
#: would be a run whose environment nobody could reproduce.
SETTINGS_PREFIX = "RESEARCH_RAG_"

#: The variable that relocates the account configuration directory. The account
#: overlay is resolved beneath it, so pointing it at an empty directory is what
#: keeps one account's settings out of a measured run.
CONFIG_HOME_ENV = "XDG_CONFIG_HOME"

#: How long a resolution is given. It reads a TOML file and coerces forty values,
#: so a longer wait means the child is not this module.
RESOLVE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True, slots=True)
class SettingsAnswer:
    """What the tree resolved, and how it says each key reached its value."""

    document: str
    values: dict[str, Any]
    baseline_layers: dict[str, str]
    overridden: tuple[str, ...]

    def describe(self) -> dict[str, Any]:
        """The record form, with the document's own digest rather than its text."""

        import hashlib

        return {
            "key_count": len(self.values),
            "document_sha256": hashlib.sha256(self.document.encode()).hexdigest(),
            "overridden": list(self.overridden),
            "layers": self.baseline_layers,
            "values": self.values,
        }


def resolve_settings_for_tree(
    engine: EngineSource,
    *,
    project: Path,
    overlay: dict[str, Any],
    config_home: Path | None = None,
) -> SettingsAnswer:
    """Resolve `project`'s settings as `engine` would, then apply `overlay`.

    The project handed in is the **source** project, not the sandbox, so the
    baseline is the behaviour a reader would get today. The sandbox inherits the
    resolved values as a pinned file, which is how a run stops depending on the
    account layer.
    """

    request = json.dumps(
        {"project": str(project), "overlay": overlay}, ensure_ascii=False
    )
    environment = child_environment(engine, config_home=config_home)
    try:
        completed = subprocess.run(
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
    return SettingsAnswer(
        document=str(answer["document"]),
        values=dict(answer["values"]),
        baseline_layers=dict(answer["baseline_layers"]),
        overridden=tuple(str(key) for key in answer["overridden"]),
    )


def child_environment(
    engine: EngineSource,
    *,
    config_home: Path | None = None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """The environment a child measuring through `engine` runs with.

    Four things are deliberate here.

    - The tree's `src` is prepended to `PYTHONPATH`, so the app and the gateway it
      spawns both import the tree rather than the installed distribution.
    - Every `RESEARCH_RAG_` variable is dropped, because the environment layer
      outranks the pinned file and a variable exported for another purpose would
      silently decide a measurement.
    - `XDG_CONFIG_HOME` points at a directory with no file in it, so the account
      overlay resolves somewhere empty. The model cache is deliberately left
      alone: those are immutable binaries, and re-deriving them would need a
      network this harness does not use.
    - Nothing else is touched, so a child inherits the machine's own paths and
      credentials for whatever the app legitimately needs.
    """

    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith(SETTINGS_PREFIX):
            del environment[name]
    existing = environment.get(ENGINE_HARNESS_ENV, "")
    tree_source = str(engine.source_relative)
    environment[ENGINE_HARNESS_ENV] = (
        tree_source if not existing else f"{tree_source}{os.pathsep}{existing}"
    )
    if config_home is not None:
        environment[CONFIG_HOME_ENV] = str(config_home)
    if extra:
        environment.update(extra)
    return environment


def _last_lines(text: str, count: int = 6) -> str:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(lines[-count:]) if lines else "(no message)"


__all__ = [
    "CONFIG_HOME_ENV",
    "RESOLVER_MODULE",
    "SETTINGS_PREFIX",
    "SettingsAnswer",
    "child_environment",
    "resolve_settings_for_tree",
]
