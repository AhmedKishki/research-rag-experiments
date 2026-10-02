"""Prove that a source project was not written to.

An experiment is worth nothing if it quietly changed the corpus it measured, so
the harness digests a source project's bytes before a run and again after it. A
run whose two digests differ is a failed run, and the report says which files
differed rather than only that something did.

Hashing everything is affordable: the corpus and its generations are about a
gigabyte, which reads in under a second, and a cheaper guard would be a guard that
missed the one write that mattered.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ExperimentError

#: What a digest covers. These are the two directories a research-rag project
#: owns beneath its root: the originals, and every piece of state derived from
#: them. The root itself is included so a file added beside them is caught.
GUARDED_ENTRIES = (".research-rag", "sources")

#: How much is read per read call. Large enough to keep the syscall count low,
#: small enough that a digest is not one enormous allocation.
BLOCK_BYTES = 1 << 20

#: A path longer than this is reported rather than joined, so a digest cannot be
#: made to overflow by a deeply nested path. Nothing the app writes is near it.
MAXIMUM_RELATIVE_PATH = 1024


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Every guarded file's path, size, and content digest, at one moment."""

    root: Path
    entries: dict[str, tuple[int, str]]
    taken_at: float
    elapsed_seconds: float

    @property
    def file_count(self) -> int:
        return len(self.entries)

    @property
    def byte_count(self) -> int:
        return sum(size for size, _ in self.entries.values())

    def digest(self) -> str:
        """One comparable value for the whole project.

        The path and the content digest are both folded in, so a rename and a
        rewrite are different changes and neither can pass as the other.
        """

        accumulator = hashlib.sha256()
        for relative in sorted(self.entries):
            size, content = self.entries[relative]
            accumulator.update(f"{relative}\0{size}\0{content}\n".encode())
        return accumulator.hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "file_count": self.file_count,
            "byte_count": self.byte_count,
            "digest": self.digest(),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def snapshot(root: Path) -> Snapshot:
    """Digest every guarded file under a project root.

    A path that is a symbolic link, a socket, or anything that is not a regular
    file is not a file the app wrote and is skipped rather than followed, so a
    link out of the project cannot make the guard read outside it.
    """

    project = Path(root).expanduser().resolve()
    if not project.is_dir():
        raise ExperimentError(f"Project root is not a directory: {project}")

    started = time.perf_counter()
    entries: dict[str, tuple[int, str]] = {}
    for entry in GUARDED_ENTRIES:
        base = project / entry
        if not base.exists():
            continue
        if base.is_symlink():
            raise ExperimentError(
                f"{base} is a symlink; a guarded path must be a real directory"
            )
        if not base.is_dir():
            raise ExperimentError(f"{base} is not a directory")
        for path in _walk(base):
            relative = path.relative_to(project).as_posix()
            if len(relative) > MAXIMUM_RELATIVE_PATH:
                raise ExperimentError(f"Guarded path is too long to record: {relative}")
            entries[relative] = _digest_file(path)
    return Snapshot(
        root=project,
        entries=entries,
        taken_at=started,
        elapsed_seconds=time.perf_counter() - started,
    )


def differences(before: Snapshot, after: Snapshot) -> list[str]:
    """What changed between two snapshots of the same project, as sorted lines.

    An empty list is the claim this harness exists to support, so it is computed
    from the file lists rather than from the two digests: a reader can see which
    file moved, not only that the total moved.
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
    return changed


def _walk(base: Path) -> list[Path]:
    """Every regular file beneath `base`, in a stable order."""

    found: list[Path] = []
    for path in base.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        found.append(path)
    return sorted(found)


def _digest_file(path: Path) -> tuple[int, str]:
    accumulator = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(BLOCK_BYTES):
            accumulator.update(block)
            size += len(block)
    return size, accumulator.hexdigest()


__all__ = [
    "BLOCK_BYTES",
    "GUARDED_ENTRIES",
    "Snapshot",
    "differences",
    "snapshot",
]
