"""Name the research-rag tree an experiment measures.

A number is only attributable to an engine, so every record names the tree, the
commit, and whether the tree was dirty when the run started. A tree that is not
a git checkout is reported with a null revision rather than refused, because a
measurement against an unversioned tree is still a measurement and the record
has to be able to say so.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ExperimentError

#: What makes a directory a research-rag tree. The harness script is named here
#: rather than imported, because it is a script the app ships in its repository
#: and does not install as a module.
HARNESS_RELATIVE = Path("scripts") / "evaluate_retrieval.py"
PACKAGE_RELATIVE = Path("src") / "research_rag" / "__init__.py"

#: The variable a child process is given to select a tree, prepended ahead of
#: the interpreter's own path. The gateway the app spawns inherits it, so a code
#: arm's change reaches the retriever as well as the driver.
ENGINE_HARNESS_ENV = "PYTHONPATH"

#: How long a git question is given before it is reported as unanswered. A git
#: call that hangs is a repository on a stalled filesystem, and a run must not
#: wait on it.
GIT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class EngineSource:
    """One research-rag tree, and what is known about where it came from."""

    root: Path
    revision: str | None
    branch: str | None
    dirty: bool | None
    untracked: int

    @property
    def harness(self) -> Path:
        """The evaluation script this tree ships."""

        return self.root / HARNESS_RELATIVE

    @property
    def source_relative(self) -> Path:
        """The directory that goes first on `PYTHONPATH` to select this tree."""

        return self.root / "src"

    def describe(self) -> dict[str, Any]:
        """The record form: what a reader needs to attribute a number."""

        return {
            "root": str(self.root),
            "revision": self.revision,
            "branch": self.branch,
            "dirty": self.dirty,
            "untracked_file_count": self.untracked,
            "harness": str(self.harness),
            "version": _installed_version(self.root),
        }


def locate_engine(root: Path) -> EngineSource:
    """Return the engine at `root`, refusing anything that is not one.

    The check is the package and the harness script, because a directory that
    has both is the tree a measurement runs against and anything else is a path
    a caller named by mistake.
    """

    tree = Path(root).expanduser().resolve()
    if not tree.is_dir():
        raise ExperimentError(f"Engine path is not a directory: {tree}")
    missing = [
        str(relative)
        for relative in (PACKAGE_RELATIVE, HARNESS_RELATIVE)
        if not (tree / relative).is_file()
    ]
    if missing:
        raise ExperimentError(
            f"Not a research-rag tree: {tree} has no {', '.join(missing)}. "
            "Pass --app-source a checkout of research-rag."
        )
    revision, branch = _git_queries(tree)
    status = _git_status(tree)
    return EngineSource(
        root=tree,
        revision=revision,
        branch=branch,
        dirty=None if status is None else bool(status["tracked"]),
        untracked=0 if status is None else int(status["untracked"]),
    )


def git_revision_of(root: Path) -> str | None:
    """The commit a directory is on, or None when it is not a git checkout.

    Used for a disposable code arm, whose record must name the base commit it
    was copied from and the diff it applied on top.
    """

    revision, _ = _git_queries(Path(root))
    return revision


def git_diff_of(root: Path, base: str) -> str | None:
    """The working-tree diff a code arm carries, or None when there is none.

    A patch file is one way to state a variant and an edited tree is another;
    this reports the second so a record names a variant however it was written.
    """

    result = _git(root, ["diff", base])
    return result if result is not None else ""


def _git(root: Path, arguments: list[str]) -> str | None:
    """Run a git question, treating a failure as an answer this tree cannot give."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_queries(root: Path) -> tuple[str | None, str | None]:
    revision = _git(root, ["rev-parse", "HEAD"])
    if revision is None:
        return None, None
    branch = _git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    return revision, None if branch == "HEAD" else branch


def _git_status(root: Path) -> dict[str, int] | None:
    """How many tracked files differ and how many are untracked, or None."""

    output = _git(root, ["status", "--porcelain"])
    if output is None:
        return None
    tracked = 0
    untracked = 0
    for line in output.splitlines():
        if line.startswith("??"):
            untracked += 1
        else:
            tracked += 1
    return {"tracked": tracked, "untracked": untracked}


def _installed_version(root: Path) -> str | None:
    """The version the tree's own package declares, or None when it declares none.

    Read from the tree rather than from the installed distribution, because the
    question is which version this tree is. A tree that has not been released
    declares nothing the registry would answer, so its `pyproject.toml` is the
    fallback rather than a second source of truth invented here.
    """

    import tomllib

    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        with pyproject.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = document.get("project")
    if not isinstance(project, dict):
        return None
    version = project.get("version")
    return str(version) if version else None


__all__ = [
    "ENGINE_HARNESS_ENV",
    "EngineSource",
    "git_diff_of",
    "git_revision_of",
    "locate_engine",
]
