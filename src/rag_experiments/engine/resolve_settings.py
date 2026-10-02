"""Resolve the settings a tree under test would use, and pin them into a file.

This module is the boundary between the harness and the app's settings. It is run
as a subprocess with the tree under test first on ``PYTHONPATH``, so
``research_rag`` resolves to that tree and the answer comes from the engine the
arm will actually run: its own registry decides which keys exist, its own
``default.toml`` supplies the defaults, and its own layer stack decides which
layer won.

The output is a document that names **every** setting the tree declares, not the
handful an arm changed. Pinning the whole set is what makes a run reproducible:
the account overlay, a shell variable, and a future change to a packaged default
cannot move a number, because none of them is above a project file that states
every value.

Usage, with a request on stdin as JSON:

    {"project": "/path/to/project", "overlay": {"retrieval.rrf_k": 120}}

The response on stdout is the document, the value each key resolved to, and the
layer the baseline value came from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

#: What the parent sends. Kept to a mapping with these two keys so a typo in a
#: request is a refusal rather than a silently empty overlay.
PROJECT_FIELD = "project"
OVERLAY_FIELD = "overlay"


class SettingsResolutionError(RuntimeError):
    """The tree's own settings machinery refused the request.

    This is a different failure from the toolkit's: the app refused an unknown
    key, an out-of-range value, or a malformed layer, and the parent reports its
    message verbatim rather than restating it.
    """


def resolve(project: Path, overlay: dict[str, Any]) -> dict[str, Any]:
    """Return the pinned document, the resolved values, and their baseline layers.

    `overlay` is applied as the command-line layer, the highest one, so an arm's
    value is the value the document carries. The baseline layers are reported
    separately because knowing that `rrf_k` came from the account overlay is part
    of what makes a number attributable.
    """

    from research_rag.project.settings import SETTINGS, sources_for
    from research_rag.project.settings_layers import SettingsError, resolve_settings

    try:
        by_field, provenance = resolve_settings(
            SETTINGS,
            sources_for(project),
            overrides=[
                f"{key}={_as_text(value)}" for key, value in sorted(overlay.items())
            ],
        )
    except SettingsError as exc:
        raise SettingsResolutionError(str(exc)) from exc

    values = {
        setting.key: by_field[setting.field]
        for setting in SETTINGS
        if setting.field in by_field
    }
    return {
        "document": _document(values),
        "values": values,
        "baseline_layers": {key: provenance.get(key, "") for key in values},
        "overridden": sorted(overlay),
    }


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
    string for every character these kinds produce. `tests/test_settings_file.py`
    reads the result back with the app's reader rather than trusting this.
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
    if not project:
        print(f"settings request has no {PROJECT_FIELD!r}", file=sys.stderr)
        return 2
    if not isinstance(overlay, dict):
        print(f"settings request {OVERLAY_FIELD!r} must be an object", file=sys.stderr)
        return 2

    try:
        answer = resolve(Path(project), overlay)
    except SettingsResolutionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(answer, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
