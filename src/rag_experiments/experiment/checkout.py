"""A disposable copy of the engine tree, for an arm that changes code.

A code arm measures engine code this repository does not have. Rather than
patching a working tree a person is editing, the arm gets its own copy of the
tree at a stated revision, applies its patch there, and runs against that. The
app source is left exactly as it was, and the record names the base revision, the
patch's own digest, and the resulting diff, so a result can be reproduced by
applying the same patch to the same commit.

The copy keeps `pyproject.toml` and `uv.lock`, so the arm's tree is the same
project; but it is not installed, because the arm does not need a second
environment. Putting the copy's `src` first on `PYTHONPATH` is enough for both
the driver and the gateway the app spawns to import it.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine.locate import EngineSource, git_revision_of
from ..errors import ExperimentError

#: What is never copied into a code arm's tree: a virtual environment would be a
#: second install of the same tree, and the caches are build products of it. The
#: repository is *not* in this list, because a patch is applied with git and a
#: revision has to be named; see `_require_repository`.
EXCLUDED = (".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache")

#: The metadata directory git keeps in a checkout.
GIT_DIRECTORY = ".git"

CHECKOUT_TIMEOUT_SECONDS = 300


@dataclass(frozen=True, slots=True)
class Checkout:
    """A tree an arm patched, and what it patched."""

    root: Path
    base_revision: str | None
    patch_path: Path | None
    patch_sha256: str | None
    resulting_diff: str
    dirty_before_patch: bool | None
    elapsed_seconds: float

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "base_revision": self.base_revision,
            "patch": None if self.patch_path is None else str(self.patch_path),
            "patch_sha256": self.patch_sha256,
            "applied_diff": self.resulting_diff,
            "dirty_before_patch": self.dirty_before_patch,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def make_checkout(
    destination: Path,
    *,
    app_source: EngineSource,
    base: str | None,
    patch: Path | None,
) -> Checkout:
    """Copy the engine tree, move it to `base`, and apply `patch`.

    A tree that is not a git checkout cannot be moved to a revision, and a patch
    is applied with git, so both are refused rather than approximated. A tree that
    is dirty is copied as it stands when no `base` is named, and the record says it
    was dirty, because an unversioned change is a variant the reader has to see.
    """

    if destination.exists():
        raise ExperimentError(
            f"A checkout already exists at {destination}. Remove it before running "
            "the arm again."
        )
    if not app_source.root.is_dir():
        raise ExperimentError(f"The app source is not a directory: {app_source.root}")
    _require_repository(app_source.root)

    started = time.perf_counter()
    shutil.copytree(
        app_source.root,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(*EXCLUDED),
    )

    before = _engine_at(destination)
    if base:
        _git(destination, ["checkout", "--detach", base])

    applied = _apply_patch(destination, patch) if patch else ""
    return Checkout(
        root=destination,
        base_revision=git_revision_of(destination),
        patch_path=patch,
        patch_sha256=_digest(patch) if patch else None,
        resulting_diff=applied,
        dirty_before_patch=before.dirty,
        elapsed_seconds=time.perf_counter() - started,
    )


def _require_repository(root: Path) -> None:
    """Refuse a tree whose git metadata is a link rather than a directory.

    A linked worktree's `.git` names a git directory outside the tree. Copying the
    link as a file would leave the copy resolving to the tree it was copied from,
    so a `checkout` or an `apply` in the copy would move the original. The
    remedy is a plain clone or a full checkout of the same commit.
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
        + ". A code arm copies the repository so its patch is applied to the copy "
        "and its revision can be named, and a linked worktree cannot be copied "
        "safely. Point --app-source at a full checkout or a clone of the same "
        "commit, or use a settings arm."
    )


def _apply_patch(tree: Path, patch: Path) -> str:
    """Apply a unified diff to a tree, and return the diff the tree now carries.

    A patch that does not apply is a run that cannot measure the variant it
    names, so it is refused with git's own message rather than skipped.
    """

    before = _diff(tree)
    completed = _git(tree, ["apply", "--whitespace=nowarn", str(patch.resolve())])
    if completed is None:
        raise ExperimentError(f"Git could not apply {patch} to {tree}")
    after = _diff(tree)
    return after if after != before else ""


def _engine_at(root: Path) -> EngineSource:
    from ..engine.locate import locate_engine

    return locate_engine(root)


def _git(root: Path, arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=CHECKOUT_TIMEOUT_SECONDS,
            check=False,
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


def _digest(path: Path) -> str:
    accumulator = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            accumulator.update(block)
    return accumulator.hexdigest()


__all__ = ["EXCLUDED", "Checkout", "make_checkout"]
