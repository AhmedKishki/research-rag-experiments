"""An experiment toolkit for research-rag retrieval.

The toolkit materializes a disposable copy of a research-rag project, runs one
or more measured arms against it, and records what each arm ran and measured.
It never writes inside the project it measures, and it proves that by comparing
a digest of that project's bytes taken before and after a run.

Each folder under `src/rag_experiments/` owns one concern. The package README
maps those concerns, and module docstrings state their local mechanisms.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

DISTRIBUTION = "rag-experiments"


def toolkit_version() -> str:
    """The version of the installed distribution, which is the version a run names."""

    try:
        return version(DISTRIBUTION)
    except PackageNotFoundError:  # pragma: no cover - a source tree with no install
        return "0.0.0+uninstalled"


__all__ = ["DISTRIBUTION", "toolkit_version"]
