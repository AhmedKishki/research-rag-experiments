"""A disposable checkout of the engine tree, for an arm that changes code.

A code arm measures engine code this repository does not have. Rather than
patching a working tree a person is editing, the arm gets its own checkout of the
same repository, applies its patch there, and runs against that.

The checkout is made with `git clone --local --no-hardlinks`, so the clone holds
its own copy of the objects and its own index. Nothing this module does writes to
the tree it read: git is given `GIT_OPTIONAL_LOCKS=0` for every question asked of
the source, and the commands that write run only inside the clone.

What the source tree held at the time is captured rather than assumed. The base
revision, the full diff of tracked files against it, and a digest for every
untracked file are all written into the run's own directory, and a tree that
carried uncommitted work has that work applied to the clone when no revision was
named, because "the engine as it stands" is what a settings arm measures and a
code arm that names no base is measuring the same tree. The arm's own patch file is
copied into the run as well, so the variant can be reproduced from the record
without the specification that named it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine.locate import (
    EngineSource,
    git_diff_of,
    git_revision_of,
    git_untracked_of,
    locate_engine,
)
from ..errors import ExperimentError
from ..niceness import run_low_priority

#: What is never copied into a code arm's tree: a virtual environment would be a
#: second install of the same tree, and the caches are build products of it. The
#: repository is *not* in this list, because the clone brings its own.
EXCLUDED = (".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache")

#: The metadata directory git keeps in a checkout.
GIT_DIRECTORY = ".git"

#: The evidence a code arm's run directory carries about the tree it copied.
SOURCE_EVIDENCE = "engine-source.json"
SOURCE_DIFF = "engine-source.diff"
PATCH_COPY = "patch.diff"

#: How the copy is made, recorded so a reader knows the original's git directory
#: was never written to.
CLONE_METHOD = "git clone --local --no-hardlinks"

CHECKOUT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True, slots=True)
class Checkout:
    """A tree an arm patched, and what it patched."""

    root: Path
    base_revision: str | None
    method: str
    patch_path: Path | None
    patch_source: str
    patch_sha256: str | None
    resulting_diff: str
    dirty_before_patch: bool | None
    source_tree_dirty: bool
    source_dirty_applied: bool
    source_untracked: tuple[dict[str, Any], ...]
    evidence_directory: Path
    elapsed_seconds: float

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "base_revision": self.base_revision,
            "method": self.method,
            "patch": None if self.patch_path is None else str(self.patch_path),
            "patch_source": self.patch_source,
            "patch_sha256": self.patch_sha256,
            "applied_diff": self.resulting_diff,
            "dirty_before_patch": self.dirty_before_patch,
            "source_tree_dirty": self.source_tree_dirty,
            "source_dirty_applied": self.source_dirty_applied,
            "source_untracked": list(self.source_untracked),
            "evidence_directory": str(self.evidence_directory),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def make_checkout(
    destination: Path,
    *,
    app_source: EngineSource,
    base: str | None,
    patch: Path | None,
    evidence_directory: Path | None = None,
) -> Checkout:
    """Make an independent checkout of the engine tree, and apply what the arm asked for.

    A tree that is not a git checkout cannot be cloned, and a patch is applied
    with git, so both are refused rather than approximated. A tree whose git
    metadata is a linked worktree's file is refused for the same reason: cloning it
    is possible, but the revision it names is not the revision the working copy
    holds, so the number would be attributed to the wrong code.
    """

    if destination.exists() or destination.is_symlink():
        raise ExperimentError(
            f"A checkout already exists at {destination}. Remove it before running "
            "the arm again."
        )
    if not app_source.root.is_dir():
        raise ExperimentError(f"The app source is not a directory: {app_source.root}")
    _require_repository(app_source.root)
    evidence = Path(evidence_directory) if evidence_directory else destination.parent

    started = time.perf_counter()
    source_revision = git_revision_of(app_source.root)
    tracked_diff = git_diff_of(app_source.root, "HEAD") if source_revision else ""
    untracked = git_untracked_of(app_source.root)
    untracked_evidence = _untracked_evidence(app_source.root, untracked)
    source_dirty = bool(tracked_diff) or bool(untracked)
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / SOURCE_EVIDENCE).write_text(
        json.dumps(
            {
                "source_root": str(app_source.root),
                "base_requested": base,
                "source_revision": source_revision,
                "dirty": source_dirty,
                "tracked_diff_bytes": len(tracked_diff),
                "untracked": untracked_evidence,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if tracked_diff:
        (evidence / SOURCE_DIFF).write_text(tracked_diff, encoding="utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    _clone(app_source.root, destination)
    if base:
        _git(destination, ["checkout", "--detach", base])

    applied_dirty = False
    if source_dirty and not base:
        # No revision was named, so the arm measures the tree as it stands, which
        # is the tree a settings arm would have measured.
        if tracked_diff:
            _git(
                destination,
                ["apply", "--whitespace=nowarn", str(evidence / SOURCE_DIFF)],
            )
        _restore_untracked(app_source.root, destination, untracked)
        applied_dirty = True

    patch_copy: Path | None = None
    patch_digest: str | None = None
    if patch:
        patch_copy = evidence / PATCH_COPY
        shutil.copy2(patch.resolve(), patch_copy)
        patch_digest = _digest(patch_copy)
        _git(destination, ["apply", "--whitespace=nowarn", str(patch_copy.resolve())])

    before = _dirty(destination)
    return Checkout(
        root=destination,
        base_revision=git_revision_of(destination),
        method=CLONE_METHOD,
        patch_path=patch_copy,
        patch_source="" if patch is None else str(patch),
        patch_sha256=patch_digest,
        resulting_diff=_diff(destination),
        dirty_before_patch=before,
        source_tree_dirty=source_dirty,
        source_dirty_applied=applied_dirty,
        source_untracked=untracked_evidence,
        evidence_directory=evidence,
        elapsed_seconds=time.perf_counter() - started,
    )


def _require_repository(root: Path) -> None:
    """Refuse a tree whose git metadata is a link rather than a directory.

    A linked worktree's `.git` is a file naming a git directory outside the tree,
    so its revision and its uncommitted state belong to another checkout. Cloning
    it would attribute a number to a commit the working copy is not on.
    """

    metadata = root / GIT_DIRECTORY
    if metadata.is_dir():
        return
    raise ExperimentError(
        f"{root} has no git repository of its own: {metadata} is "
        + (
            "a file, so this tree is a linked worktree"
            if metadata.is_file()
            else "absent"
        )
        + ". A code arm clones the repository so its patch is applied to the clone "
        "and its revision can be named, and a linked worktree cannot be cloned "
        "safely. Point --app-source at a full checkout or a clone of the same "
        "commit, or use a settings arm."
    )


def _clone(source: Path, destination: Path) -> None:
    """Clone a local repository without linking its objects to the original's.

    `--local` would otherwise hard-link the object store, which means the clone
    shares storage with the developer's checkout: a `git gc` or a repack in one
    changes the bytes the other reads.
    """

    _git(
        destination.parent,
        [
            "clone",
            "--local",
            "--no-hardlinks",
            "--quiet",
            str(source),
            str(destination),
        ],
        cwd=destination.parent,
    )


def _untracked_evidence(root: Path, names: list[str]) -> tuple[dict[str, Any], ...]:
    """A digest for every file git holds no record of, refusing a link among them.

    An untracked symbolic link has no content to copy into the clone, and copying
    its target instead would put a file in the clone that the tree does not have.
    """

    evidence: list[dict[str, Any]] = []
    for name in names:
        path = root / name
        if path.is_symlink():
            raise ExperimentError(
                f"{path} is an untracked symbolic link. An arm cannot reproduce a "
                "tree holding one, so it is refused rather than approximated."
            )
        if not path.is_file():
            continue
        evidence.append(
            {
                "path": name,
                "byte_count": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    return tuple(evidence)


def _restore_untracked(source: Path, destination: Path, names: list[str]) -> None:
    """Copy the source tree's untracked files into the clone, byte for byte."""

    for name in names:
        origin = source / name
        if not origin.is_file() or origin.is_symlink():
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)


def _git(root: Path, arguments: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = run_low_priority(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=CHECKOUT_TIMEOUT_SECONDS,
            check=False,
            cwd=None if cwd is None else str(cwd),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExperimentError(
            f"Git {' '.join(arguments)} failed in {root}: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise ExperimentError(
            f"git {' '.join(arguments)} failed in {root}:\n{completed.stderr.strip()}"
        )
    return completed.stdout


def _diff(root: Path) -> str:
    return _git(root, ["diff", "HEAD"])


def _dirty(root: Path) -> bool | None:
    """Whether a checkout differs from its own commit at all, or None if git cannot say.

    Both a modified tracked file and an added untracked one count: a patch that adds
    a module has changed what the engine is, and a record reporting only tracked
    changes would call that checkout clean.
    """

    try:
        engine = locate_engine(root)
    except ExperimentError:
        return None
    if engine.dirty is None:
        return None
    return bool(engine.dirty) or bool(engine.untracked)


def _digest(path: Path) -> str:
    accumulator = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            accumulator.update(block)
    return accumulator.hexdigest()


__all__ = [
    "CLONE_METHOD",
    "EXCLUDED",
    "PATCH_COPY",
    "SOURCE_DIFF",
    "SOURCE_EVIDENCE",
    "Checkout",
    "make_checkout",
]
