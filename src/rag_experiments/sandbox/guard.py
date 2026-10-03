"""Prove that a source project was not written to.

An experiment is worth nothing if it quietly changed the corpus it measured, so
the harness digests a source project's bytes before a run and again after it. A
run whose two digests differ is a failed run, and the report says which paths
differ rather than only that something did.

What is covered is the whole project: its originals at whatever path its
descriptor names, the review state beside them, every generation, and every file
at the root. A denylist of the two directories the app owns would leave a write
to any other path unguarded, and a project's source directory is configurable, so
a name spelled here would also be a name that can be wrong.

What is not covered is the state a serving app rewrites on its own: the pid, the
port, the terminal, the project lock, the logs, and the gateway's own runtime.
Excluding those is what lets a run measure a project that is up. Everything else
a change to is named, and a change made by the app or by a person while the run
was in progress disqualifies the run, because the corpus the arms read is then not
the corpus on disk.

Hashing everything is affordable: the corpus and its generations are about a
gigabyte, which reads in under a second, and a cheaper guard would be a guard that
missed the one write that mattered.
"""

from __future__ import annotations

import hashlib
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._imports import without_bytecode
from ..errors import ExperimentError
from .layout import relocated_runtime_of

#: How much is read per read call. Large enough to keep the syscall count low,
#: small enough that a digest is not one enormous allocation.
BLOCK_BYTES = 1 << 20

#: A path longer than this is reported rather than joined, so a digest cannot be
#: made to overflow by a deeply nested path. Nothing the app writes is near it.
MAXIMUM_RELATIVE_PATH = 1024

#: What a path is recorded as. A directory is recorded so that removing an empty
#: one is a change rather than an absence, and a symbolic link is recorded by the
#: string it points at rather than followed, so a link out of the project cannot
#: make the guard read outside it or make its content part of the digest.
FILE = "file"
DIRECTORY = "directory"
LINK = "link"
OTHER = "other"

#: Where the paths of a relocated runtime directory are keyed, so its generations
#: read as their own tree rather than colliding with paths under the project root.
RELOCATED_PREFIX = "relocated-runtime/"


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Every guarded path's kind, size, and content digest, at one moment."""

    root: Path
    entries: dict[str, tuple[str, int, str]]
    volatile: tuple[str, ...]
    relocated_root: Path | None
    registry_path: Path | None
    registry_digest: str | None
    unreadable: tuple[str, ...]
    taken_at: float
    elapsed_seconds: float

    @property
    def file_count(self) -> int:
        return sum(1 for kind, _size, _digest in self.entries.values() if kind == FILE)

    @property
    def byte_count(self) -> int:
        return sum(
            size for kind, size, _digest in self.entries.values() if kind == FILE
        )

    @property
    def path_count(self) -> int:
        return len(self.entries)

    def digest(self) -> str:
        """One comparable value for the whole project.

        The path, the kind, and the content digest are all folded in, so a rename,
        a rewrite, and a directory that became a file are different changes and
        none can pass as another.
        """

        accumulator = hashlib.sha256()
        for relative in sorted(self.entries):
            kind, size, content = self.entries[relative]
            accumulator.update(f"{relative}\0{kind}\0{size}\0{content}\n".encode())
        return accumulator.hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "file_count": self.file_count,
            "path_count": self.path_count,
            "byte_count": self.byte_count,
            "digest": self.digest(),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "excluded_paths": list(self.volatile),
            "relocated_runtime": (
                None if self.relocated_root is None else str(self.relocated_root)
            ),
            "unreadable": list(self.unreadable),
            "account_registry": {
                "path": (
                    None if self.registry_path is None else str(self.registry_path)
                ),
                "sha256": self.registry_digest,
            },
        }


@without_bytecode
def volatile_paths() -> tuple[str, ...]:
    """The paths a serving app rewrites on its own, named by the app's own code.

    These are excluded from the digest and listed in the record, so a reader can
    see exactly what the guard did not watch. The names come from the app: its pid,
    port, terminal and lock files, its log directory, and the gateway's runtime
    directory. Everything else under a project is watched.
    """

    from research_rag.project import state_files

    state_root = f"{state_files.PORTABLE_DIRECTORY}/{state_files.RUNTIME_DIRECTORY}"
    return tuple(f"{state_root}/{name}" for name in _volatile_names())


def _volatile_names() -> tuple[str, ...]:
    """The excluded names as the app's own state root spells them.

    They are stated relative to the runtime directory rather than to the project,
    so the same names apply to a project whose state was relocated.
    """

    from research_rag.project import config as app_config
    from research_rag.project import state_files

    return (
        state_files.PID_FILE,
        state_files.PORT_FILE,
        state_files.TTY_FILE,
        state_files.LOCK_FILE,
        _literal_of(app_config.ResearchConfig, "logs_root"),
        _literal_of(app_config.ResearchConfig, "ultrarag_workspace"),
    )


@without_bytecode
def snapshot(root: Path) -> Snapshot:
    """Digest every guarded path under a project root.

    The project root itself is the base, so a project whose originals are not in a
    directory this harness knows is still covered. A project's relocated runtime
    directory is digested too, because a change to a generation is a change to the
    corpus whether it was written beside the project or elsewhere.
    """

    project = Path(root).expanduser().resolve()
    if not project.is_dir():
        raise ExperimentError(f"Project root is not a directory: {project}")

    started = time.perf_counter()
    volatile = volatile_paths()
    names = frozenset(_volatile_names())
    entries: dict[str, tuple[str, int, str]] = {}
    unreadable: list[str] = []
    _scan(project, "", volatile, entries, unreadable)

    relocated = relocated_runtime_of(project)
    if relocated is not None and relocated.is_dir():
        # A relocated state root carries the same names, so the exclusions apply to
        # it under the same paths the project itself uses.
        _scan(relocated, RELOCATED_PREFIX, names, entries, unreadable)

    registry_path, registry_digest = _account_registry()
    return Snapshot(
        root=project,
        entries=entries,
        volatile=volatile,
        relocated_root=relocated,
        registry_path=registry_path,
        registry_digest=registry_digest,
        unreadable=tuple(sorted(unreadable)),
        taken_at=started,
        elapsed_seconds=time.perf_counter() - started,
    )


def differences(before: Snapshot, after: Snapshot) -> list[str]:
    """What changed between two snapshots of the same project, as sorted lines.

    An empty list is the claim this harness exists to support, so it is computed
    from the path lists rather than from the two digests: a reader can see which
    path moved, not only that the total moved. A difference in what the account's
    project registry holds is included, because a sandbox that registered itself
    would have evicted the original's record.
    """

    if before.root != after.root:
        return [f"root changed: {before.root} -> {after.root}"]
    changed: list[str] = []
    for relative in sorted(set(before.entries) | set(after.entries)):
        was = before.entries.get(relative)
        now = after.entries.get(relative)
        if was == now:
            continue
        if was is None:
            changed.append(f"added: {relative}")
        elif now is None:
            changed.append(f"removed: {relative}")
        else:
            changed.append(f"changed: {relative}")
    if before.registry_digest != after.registry_digest:
        where = after.registry_path or before.registry_path
        changed.append(f"account registry changed: {where}")
    return changed


def _scan(
    base: Path,
    prefix: str,
    excluded_paths: tuple[str, ...] | frozenset[str],
    entries: dict[str, tuple[str, int, str]],
    unreadable: list[str],
) -> None:
    """Walk one root without following a symbolic link out of it.

    `os.scandir` rather than `rglob`, because a recursive glob follows a
    directory symlink on the versions of Python this package supports, and a link
    out of the project would then be read as if it were part of the corpus.
    """

    excluded = frozenset(excluded_paths)
    # Each frame carries the directory and the key prefix it contributes, so a
    # nested path is recorded with every name between it and the root.
    stack: list[tuple[Path, str]] = [(base, prefix)]
    while stack:
        directory, prefix_here = stack.pop()
        try:
            with os.scandir(directory) as found:
                listing = sorted(found, key=lambda item: item.name)
        except OSError as exc:
            unreadable.append(f"{directory}: {exc}")
            continue
        for entry in listing:
            key = f"{prefix_here}{entry.name}"
            if len(key) > MAXIMUM_RELATIVE_PATH:
                raise ExperimentError(f"Guarded path is too long to record: {key}")
            if key in excluded:
                continue
            try:
                if entry.is_symlink():
                    target = os.readlink(entry.path)
                    entries[key] = (LINK, len(target), _digest_text(target))
                elif entry.is_dir(follow_symlinks=False):
                    entries[key] = (DIRECTORY, 0, "")
                    stack.append((Path(entry.path), f"{key}/"))
                elif entry.is_file(follow_symlinks=False):
                    size, content = _digest_file(Path(entry.path))
                    entries[key] = (FILE, size, content)
                else:
                    kind = _other_kind(entry)
                    entries[key] = (OTHER, 0, _digest_text(kind))
            except OSError as exc:
                # A path that vanished or was refused mid-walk is reported rather
                # than skipped: a guard that cannot see a file cannot say it was
                # not written to.
                unreadable.append(f"{key}: {exc}")


def _other_kind(entry: os.DirEntry[str]) -> str:
    try:
        return stat.filemode(entry.stat(follow_symlinks=False).st_mode)
    except OSError:
        return "unknown"


def _account_registry() -> tuple[Path | None, str | None]:
    """The account's project registry, digested because a copy must never join it.

    The path is the app's own. A registry that does not exist is None rather than
    an empty digest, so a machine that has never registered a project is not
    reported as having one whose content moved.
    """

    from research_rag.project.registry import registry_path

    path = Path(registry_path()).expanduser()
    if not path.is_file():
        return path, None
    return path, _digest_file(path)[1]


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _digest_file(path: Path) -> tuple[int, str]:
    accumulator = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(BLOCK_BYTES):
            accumulator.update(block)
            size += len(block)
    return size, accumulator.hexdigest()


def _literal_of(owner: type, property_name: str) -> str:
    """The one path literal inside an app property, read from its code."""

    getter = getattr(getattr(owner, property_name, None), "fget", None)
    if getter is None:
        raise ExperimentError(
            f"The app's ResearchConfig no longer has a {property_name} property, so "
            "this harness cannot say which directory a serving app rewrites."
        )
    literals = [value for value in getter.__code__.co_consts if isinstance(value, str)]
    if len(literals) != 1:
        raise ExperimentError(
            f"The app's ResearchConfig.{property_name} holds {len(literals)} string "
            f"literals, so this harness cannot say which one is the name: {literals}"
        )
    return literals[0]


__all__ = [
    "BLOCK_BYTES",
    "DIRECTORY",
    "FILE",
    "LINK",
    "OTHER",
    "RELOCATED_PREFIX",
    "Snapshot",
    "differences",
    "snapshot",
    "volatile_paths",
]
