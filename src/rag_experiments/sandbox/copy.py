"""Materialize a disposable copy of a project, and destroy one this harness made.

The copy is a real research-rag project: the same originals under the same
source-relative paths, the same review state, and a selected generation. That is
what lets the app's own harness measure it without knowing it is a copy, and what
makes a re-ingestion in a sandbox produce the same `source_id` values the
original would.

Every file is copied as bytes into a new inode. A hard link, a reflink, or a
bind mount would make a write inside the sandbox a write to the corpus, and the
guard would then report a change it caused rather than one it found.

Two things are deliberately not carried, and neither is named by hand:

- Everything under the runtime directory except the selected-generation pointer
  and the generations a caller asked for. That covers the process state, the
  project lock, a build in progress, and the build journal, by construction.
- The runtime pointer that names a relocated runtime directory. It is
  machine-local, and a copy carrying it would read the original's generations.

The account's project registry is never touched, because a copy keeps the
original's `project_id` and registering it would evict the original's record. A
sandbox is addressed by path and is never registered.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import ExperimentError
from .layout import (
    ProjectLayout,
    existing_generations,
    read_layout,
    refuse_nested,
    selected_generation,
)

#: The file inside a workspace that says what this harness made. It sits beside
#: the project rather than inside it, so a sandbox's project root holds nothing
#: but what the app itself writes and the sandbox is a valid project as it stands.
SANDBOX_RECORD = "sandbox.json"

#: The directory inside a workspace that is the project root.
PROJECT_DIRECTORY = "project"

#: The empty configuration home a child is pointed at, so the account overlay
#: resolves somewhere with no file in it. The model cache is not relocated: those
#: are immutable binaries and a run does not download anything, and its resolved
#: path is pinned into the settings a run writes.
CONFIG_HOME_DIRECTORY = "xdg"

#: Written with the same rule the app uses for its own records: readable, stable,
#: and one trailing newline.
RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Sandbox:
    """One materialized copy, and what it was made from."""

    name: str
    workspace: Path
    root: Path
    source_root: Path
    project_id: str
    generation_ids: tuple[str, ...]
    settings_file: Path
    config_home: Path
    created_at: str
    byte_count: int
    file_count: int
    elapsed_seconds: float
    origin: dict[str, Any] = field(default_factory=dict)

    @property
    def record_path(self) -> Path:
        return self.workspace / SANDBOX_RECORD

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            # Where the sandbox area is, so `sandbox remove` can be aimed at it by
            # hand once a run has finished with it.
            "workspace": str(self.workspace),
            # The copy's own originals, not the original's. A record that named
            # the original here would point a reader, and a prepare command's
            # `{source}`, at the corpus the run promised not to touch.
            "source_root": str(self.source_root),
            "original_source_root": str(self.origin.get("original_source_root") or ""),
            "project_id": self.project_id,
            "generation_ids": list(self.generation_ids),
            "settings_file": str(self.settings_file),
            "config_home": str(self.config_home),
            "created_at": self.created_at,
            "byte_count": self.byte_count,
            "file_count": self.file_count,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "origin": self.origin,
        }


def create(
    workspace: Path,
    *,
    name: str,
    source: Path,
    generations: tuple[str, ...] = (),
) -> Sandbox:
    """Copy a project into a workspace under `name` and return what was made.

    `generations` names which generation directories travel. Empty means the one
    the project's own pointer selects, which is what a measurement of today's
    behaviour needs; naming a second one is how a baseline and a trial are
    compared without either being rebuilt.
    """

    _require_segment(name, what="A sandbox name")
    refuse_nested(
        Path(workspace).expanduser().resolve() / name,
        Path(source).expanduser().resolve(),
        what="The sandbox directory",
        inside="source project",
    )
    layout = read_layout(source)
    chosen = tuple(generations) or (selected_generation(layout),)
    for generation_id in chosen:
        if not layout.generation_root(generation_id).is_dir():
            available = ", ".join(existing_generations(layout)) or "none"
            raise ExperimentError(
                f"{source} has no generation {generation_id!r}. Available: {available}."
            )

    area = (Path(workspace).expanduser().resolve()) / name
    if area.exists() or area.is_symlink():
        raise ExperimentError(
            f"A sandbox already exists at {area}. Remove it with "
            f"`rag-experiments sandbox remove --workspace {workspace} --name {name}`."
        )

    started = time.perf_counter()
    area.mkdir(parents=True)
    project_root = area / PROJECT_DIRECTORY
    try:
        copied_source = _copy_directory(
            layout.source_root,
            project_root / layout.source_relative(layout.source_root),
        )
        _copy_portable(layout, project_root)
        _copy_generations(layout, project_root, chosen)
        config_home = area / CONFIG_HOME_DIRECTORY
        config_home.mkdir()
        settings_file = project_root / layout.settings_relative
        byte_count, file_count = _measure(project_root)
        sandbox = Sandbox(
            name=name,
            workspace=area,
            root=project_root,
            source_root=copied_source,
            project_id=_project_id(layout),
            generation_ids=chosen,
            settings_file=settings_file,
            config_home=config_home,
            created_at=_now(),
            byte_count=byte_count,
            file_count=file_count,
            elapsed_seconds=time.perf_counter() - started,
            origin={
                "source_project": str(layout.root),
                "original_source_root": str(layout.source_root),
                "layout": layout.describe(),
                "pointed_generation": selected_generation(layout),
            },
        )
        _write_record(sandbox)
    except Exception:
        # A half-built sandbox is not a project, and leaving one would be a
        # directory a later run could mistake for it. Everything after the
        # directory is created is inside this, so no partial sandbox survives.
        _discard(area)
        raise
    return sandbox


def pin_settings(sandbox: Sandbox, document: str) -> Path:
    """Write a fully-resolved settings file into a sandbox, replacing any copy.

    The file names every setting the engine declared, so a run cannot be moved by
    the account overlay, an environment variable, or a later packaged default.
    """

    sandbox.settings_file.parent.mkdir(parents=True, exist_ok=True)
    sandbox.settings_file.write_text(document, encoding="utf-8")
    return sandbox.settings_file


def list_sandboxes(workspace: Path) -> list[dict[str, Any]]:
    """Every sandbox in a workspace, as the records this harness wrote.

    A directory in a workspace with no record is not reported and is not a sandbox
    this harness can remove: a marker is what makes destruction safe. A workspace
    holding run directories is searched to any depth, because a run keeps its
    sandboxes under a directory named for the run.
    """

    area = Path(workspace).expanduser().resolve()
    if not area.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for record in sorted(area.rglob(SANDBOX_RECORD)):
        if not record.is_file():
            continue
        document = json.loads(record.read_text(encoding="utf-8"))
        if isinstance(document, dict):
            found.append({**document, "workspace": str(record.parent)})
    return found


def remove(workspace: Path, name: str) -> Path:
    """Delete a sandbox this harness made, refusing anything it did not."""

    _require_segment(name, what="A sandbox name")
    area = (Path(workspace).expanduser().resolve()) / name
    if area.is_symlink():
        raise ExperimentError(
            f"{area} is a symbolic link. This harness removes a sandbox it made "
            "under a name it chose, and follows no link out of the workspace."
        )
    record = area / SANDBOX_RECORD
    if not record.is_file():
        raise ExperimentError(
            f"{area} is not a sandbox this harness made: it has no {SANDBOX_RECORD}. "
            "Nothing was removed."
        )
    try:
        stated = str(json.loads(record.read_text(encoding="utf-8")).get("name") or "")
    except (OSError, json.JSONDecodeError):
        stated = ""
    if stated != name:
        raise ExperimentError(
            f"{record} names a sandbox called {stated!r}, not {name!r}. Nothing was "
            "removed."
        )
    shutil.rmtree(area)
    return area


def _require_segment(name: str, *, what: str) -> None:
    """Refuse a name that is not one plain path segment.

    A name that reaches a parent, a root, or another directory is a name a
    `remove` would act on somewhere the caller did not name, so it is refused
    before the path is built rather than after something is created there.
    """

    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ExperimentError(
            f"{what} is a single plain path segment, not a path: {name!r}"
        )
    if os.sep in name or name.strip() != name:
        raise ExperimentError(
            f"{what} has characters a path segment cannot hold: {name!r}"
        )


def _copy_directory(source: Path, destination: Path) -> Path:
    """Copy a directory tree byte for byte, refusing what cannot be copied safely.

    A symbolic link is refused rather than followed or recreated: the app refuses
    a symlinked original and a symlinked settings file, so a sandbox holding one
    would be a project the app cannot measure. Every other entry is copied as
    bytes into a new inode, which is what keeps a write inside the copy from
    reaching the original.
    """

    if not source.is_dir():
        raise ExperimentError(f"{source} is not a directory; nothing to copy")
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ExperimentError(
                f"{path} is a symlink, and the app refuses a symlinked original. "
                "Copy it into the sandbox as a regular file, or exclude it."
            )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False, dirs_exist_ok=True)
    _assert_independent(source, destination)
    return destination


def _assert_independent(source: Path, destination: Path) -> None:
    """Refuse a copy that shares an inode with the tree it came from.

    A reflinked or hard-linked copy would pass every byte comparison and still let
    a write inside the sandbox rewrite the original, so the property is checked
    rather than assumed from the copy function used.
    """

    originals = {path.stat().st_ino for path in source.rglob("*") if path.is_file()}
    if not originals:
        return
    copies = {path.stat().st_ino for path in destination.rglob("*") if path.is_file()}
    shared = originals & copies
    if shared:
        raise ExperimentError(
            f"{destination} shares {len(shared)} inode(s) with {source}. A copy that "
            "shares storage is not isolated, so it was removed rather than measured."
        )


def _copy_portable(layout: ProjectLayout, project_root: Path) -> None:
    """Copy the portable review state, and nothing derived.

    The runtime directory is a child of the portable one, so the copy is made
    entry by entry rather than by recursing: recursing would carry all nine
    generations of the original when the run asked for one, and the exclusion
    would have to be a list of everything derived rather than one name.
    """

    destination = project_root / layout.portable_directory
    destination.mkdir(parents=True, exist_ok=True)
    excluded = {layout.runtime_directory, layout.runtime_pointer}
    for path in sorted(layout.portable_root.iterdir()):
        if path.is_symlink():
            raise ExperimentError(
                f"{path} is a symlink. A symlink in the review state would either be "
                "refused by the app or silently change what a sandbox holds, so the "
                "copy stops rather than guess which of the two it is."
            )
        if path.name in excluded:
            continue
        if path.is_dir():
            _copy_directory(path, destination / path.name)
        elif path.is_file():
            shutil.copy2(path, destination / path.name)
    (destination / layout.runtime_directory).mkdir(parents=True, exist_ok=True)


def _copy_generations(
    layout: ProjectLayout,
    project_root: Path,
    generation_ids: tuple[str, ...],
) -> None:
    """Copy the named generations and the pointer that selects one of them."""

    state = project_root / layout.portable_directory / layout.runtime_directory
    state.mkdir(parents=True, exist_ok=True)
    for name in layout.empty_runtime_directories:
        (state / name).mkdir(parents=True, exist_ok=True)
    generations = state / layout.generations_directory
    generations.mkdir(parents=True, exist_ok=True)

    for generation_id in generation_ids:
        _copy_directory(
            layout.generation_root(generation_id), generations / generation_id
        )
    pointer = layout.state_root / layout.current_pointer
    if pointer.is_file():
        (state / layout.current_pointer).write_bytes(pointer.read_bytes())


def _project_id(layout: ProjectLayout) -> str:
    document = json.loads(
        (layout.portable_root / layout.descriptor).read_text(encoding="utf-8")
    )
    return str(document.get("project_id") or "")


def _measure(root: Path) -> tuple[int, int]:
    byte_count = 0
    file_count = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            file_count += 1
            byte_count += path.stat().st_size
    return byte_count, file_count


def _write_record(sandbox: Sandbox) -> None:
    payload = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "toolkit_digest": hashlib.sha256(
            json.dumps(sandbox.describe(), sort_keys=True).encode()
        ).hexdigest(),
        **sandbox.describe(),
    }
    sandbox.record_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _discard(area: Path) -> None:
    """Remove a directory this harness created, never following a link out of it."""

    if area.is_symlink():
        area.unlink()
        return
    shutil.rmtree(area, ignore_errors=True)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


__all__ = [
    "CONFIG_HOME_DIRECTORY",
    "PROJECT_DIRECTORY",
    "RECORD_SCHEMA_VERSION",
    "SANDBOX_RECORD",
    "Sandbox",
    "create",
    "list_sandboxes",
    "pin_settings",
    "remove",
]
