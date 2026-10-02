"""Resolve the settings a tree under test would use, and pin them into a file.

This module is the boundary between the harness and the app's settings. It is run
as a subprocess with the tree under test first on ``PYTHONPATH``, so
``research_rag`` resolves to that tree and the answer comes from the engine the
arm will actually run: its own registry decides which keys exist, its own
``default.toml`` supplies the defaults, and its own layer stack decides which
layer won.

Two resolutions happen, because they answer different questions. The baseline
resolution applies no overrides at all, so its layer for every key is where that
key's value came from before an arm touched anything. The arm's resolution then
applies the overlay, and its layers show what moved. Reporting the arm's layers as
the baseline would name the command line for every setting the arm overrode, which
is true of the arm and says nothing about what the arm was compared against.

The environment layer is dropped in both (`environ={}`), so an exported variable
cannot reach the pinned file, and the account overlay is included, so the
baseline is what a reader would get today.

The output is a document that names **every** setting the tree declares, not the
handful an arm changed. Pinning the whole set is what makes a run reproducible:
the account overlay, a shell variable, and a future change to a packaged default
cannot move a number, because none of them is above a project file that states
every value.

Usage, with a request on stdin as JSON:

    {"project": "/path/to/project", "overlay": {"retrieval.rrf_k": 120},
     "pin_project": "/path/to/sandbox"}

``project`` is read and never written. ``pin_project`` is the sandbox the values
are pinned into, used once to resolve the model cache the app would open there.

The response on stdout is the document, the value each key resolved to, the
baseline values and layers, and the cache path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

#: What the parent sends. Kept to a mapping with these keys so a typo in a
#: request is a refusal rather than a silently empty overlay.
PROJECT_FIELD = "project"
OVERLAY_FIELD = "overlay"
PIN_PROJECT_FIELD = "pin_project"

#: The setting key that names the shared model-binary cache, read from the app's
#: own registry rather than spelled here as a dotted name.
MODEL_CACHE_KEY = "runtime.model_cache_root"


class SettingsResolutionError(RuntimeError):
    """The tree's own settings machinery refused the request.

    This is a different failure from the toolkit's: the app refused an unknown
    key, an out-of-range value, or a malformed layer, and the parent reports its
    message verbatim rather than restating it.
    """


def resolve(
    project: Path,
    overlay: dict[str, Any],
    pin_project: Path | None = None,
) -> dict[str, Any]:
    """Return the pinned document, the resolved values, and their baseline layers.

    `overlay` is applied as the command-line layer, the highest one, so an arm's
    value is the value the document carries. The baseline layers are reported for
    the resolution that carried no overlay, because knowing that `rrf_k` came from
    the account overlay is part of what makes a number attributable.
    """

    from research_rag.project.settings import SETTINGS, sources_for

    baseline = _resolve(
        sources_for(project), [], source=str(project), keys={s.key for s in SETTINGS}
    )
    values = dict(baseline["values"])
    layers = dict(baseline["provenance"])
    if overlay:
        applied = _resolve(
            sources_for(project),
            [f"{key}={_as_text(value)}" for key, value in sorted(overlay.items())],
            source=str(project),
            keys={s.key for s in SETTINGS},
        )
        values = dict(applied["values"])
        layers = dict(applied["provenance"])
    cache = _model_cache_root(pin_project or project, values.get(MODEL_CACHE_KEY, ""))
    values[MODEL_CACHE_KEY] = cache
    return {
        "document": _document(values),
        "values": values,
        "baseline_values": dict(baseline["values"]),
        "baseline_layers": dict(baseline["provenance"]),
        "arm_layers": layers,
        "overridden": sorted(overlay),
        "model_cache_root": cache,
    }


def _resolve(
    sources: Any, overrides: list[str], *, source: str, keys: set[str]
) -> dict[str, Any]:
    """One resolution through the app's own registry and layer stack.

    `environ={}` is the whole point of the second argument: the environment layer
    is refused, so a value exported in the reader's shell cannot reach a pinned
    file, and a baseline means the same thing on every machine.
    """

    from research_rag.project.settings import SETTINGS
    from research_rag.project.settings_layers import SettingsError, resolve_settings

    try:
        by_field, provenance = resolve_settings(
            SETTINGS, sources, overrides=overrides, environ={}
        )
    except SettingsError as exc:
        raise SettingsResolutionError(str(exc)) from exc
    values = {
        setting.key: by_field[setting.field]
        for setting in SETTINGS
        if setting.field in by_field and setting.key in keys
    }
    return {
        "values": values,
        "provenance": {key: provenance.get(key, "") for key in values},
        "source": source,
    }


def _model_cache_root(pin_project: Path, declared: Any) -> str:
    """Where the engine under test would open the shared model binaries.

    An empty setting means the per-user cache for this app, which is a path built
    from the account's own directories. A sandbox is given an empty configuration
    home, so a relative or empty value would leave the copy looking for binaries
    in a directory the copy made and would download what a run must not download.
    The app's own configuration is therefore asked where the cache is, and the
    answer is pinned as an absolute path.
    """

    from research_rag.project import config as app_config

    if declared:
        try:
            return str(Path(str(declared)).expanduser().resolve())
        except OSError as exc:
            raise SettingsResolutionError(
                f"{MODEL_CACHE_KEY} names a path that cannot be resolved: "
                f"{declared!r}: {exc}"
            ) from exc
    try:
        # The sandbox's own descriptor names where its originals are, so the app is
        # asked rather than assuming the packaged default: a project whose
        # descriptor names another directory would otherwise be refused here for
        # disagreeing with itself.
        resolved = app_config.resolve_config(
            pin_project,
            source_directory=app_config.configured_source_directory(pin_project),
            environ={},
        )
    except Exception as exc:
        raise SettingsResolutionError(
            f"The engine under test could not say where its model cache is for "
            f"{pin_project}: {exc}. Pin {MODEL_CACHE_KEY} in the arm's overlay to "
            f"an absolute path so a sandbox reads the same binaries this machine "
            f"already holds."
        ) from exc
    return str(Path(resolved.model_cache_root).expanduser().resolve())


def _as_text(value: Any) -> str:
    """Render an overlay value the way a command line would carry it.

    A bool is spelled out because `str(True)` is `True`, which no boolean
    coercion in the app's layer stack accepts; a float keeps enough digits to
    round-trip, and anything else is left to the app's own coercion.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return str(value)


def _document(values: dict[str, Any]) -> str:
    """Render the pinned values as a settings file the app's reader accepts.

    A settings file is TOML, and this toolkit ships no writer for it. The value
    domain is the app's own setting kinds, so each value is a basic string,
    integer, float, or boolean, and a JSON basic string is also a TOML basic
    string for every character these kinds produce. `tests/test_engine.py` reads
    the result back with the app's reader rather than trusting this.
    """

    sections: dict[str, list[tuple[str, str]]] = {}
    for key, value in values.items():
        section, _, name = key.partition(".")
        sections.setdefault(section, []).append((name, _literal(value)))

    lines = [
        "# Written by rag-experiments. Every setting the engine under test declares",
        "# is named here, so an arm's measurement cannot be moved by the account",
        "# overlay, an environment variable, or a later change to a packaged",
        "# default. An arm's overlay is already applied to these values.",
        "",
    ]
    for section in sorted(sections):
        lines.append(f"[{section}]")
        for name, rendered in sorted(sections[section]):
            lines.append(f"{name} = {rendered}")
        lines.append("")
    return "\n".join(lines)


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise SettingsResolutionError(
        f"A setting resolved to a value TOML cannot carry: {value!r} "
        f"({type(value).__name__})"
    )


def main() -> int:
    """Read the request, answer it, and report a refusal as a single line."""

    try:
        request = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError as exc:
        print(f"settings request is not JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(request, dict):
        print("settings request must be a JSON object", file=sys.stderr)
        return 2
    project = str(request.get(PROJECT_FIELD) or "").strip()
    overlay = request.get(OVERLAY_FIELD) or {}
    pin_project = str(request.get(PIN_PROJECT_FIELD) or "").strip()
    if not project:
        print(f"settings request has no {PROJECT_FIELD!r}", file=sys.stderr)
        return 2
    if not isinstance(overlay, dict):
        print(f"settings request {OVERLAY_FIELD!r} must be an object", file=sys.stderr)
        return 2

    try:
        answer = resolve(
            Path(project), overlay, Path(pin_project) if pin_project else None
        )
    except SettingsResolutionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(answer, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
