"""The record a run leaves, and the verdict that says whether it counts.

Every run writes one JSON file. It is written whether the run succeeded or not,
because a failed run with its provenance is the most useful thing an experiment
produces: it names the engine, the corpus digest, the settings, and the exact
commands, so the failure is attributable rather than mysterious.

The guard is part of the record and part of the verdict. A run whose source
project digest moved is a failed measurement, whatever its numbers say, and this
module says so in one field a caller can read without parsing anything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import toolkit_version
from ..sandbox import GUARDED_ENTRIES, Snapshot, differences, snapshot

#: The record's own version.
RECORD_SCHEMA_VERSION = 1

#: Where a run's record and its arm directories live inside a runs directory.
RECORD_NAME = "run.json"

#: The verdicts a run can carry. `verified` means the source project's bytes were
#: identical before and after; nothing else counts as a measurement.
VERDICT_VERIFIED = "verified"
VERDICT_SOURCE_CHANGED = "source_project_changed"
VERDICT_NO_ARMS = "no_arm_measured"


@dataclass(frozen=True, slots=True)
class SourceGuard:
    """What the source project's bytes were before a run and after it."""

    before: Snapshot
    after: Snapshot
    differences: tuple[str, ...]

    @property
    def unchanged(self) -> bool:
        return not self.differences

    def verdict(self, arm_count: int) -> str:
        if not arm_count:
            return VERDICT_NO_ARMS
        return VERDICT_VERIFIED if self.unchanged else VERDICT_SOURCE_CHANGED

    def describe(self) -> dict[str, Any]:
        return {
            "guarded_entries": list(GUARDED_ENTRIES),
            "before": self.before.describe(),
            "after": self.after.describe(),
            "unchanged": self.unchanged,
            "differences": list(self.differences),
        }


def take_guard(root: Path) -> Snapshot:
    """Digest a source project before a run begins."""

    return snapshot(root)


def close_guard(before: Snapshot) -> SourceGuard:
    """Digest the same project again, and say what moved."""

    after = snapshot(before.root)
    return SourceGuard(
        before=before,
        after=after,
        differences=tuple(differences(before, after)),
    )


def write_record(
    run_directory: Path,
    *,
    name: str,
    elapsed_seconds: float,
    spec: dict[str, Any],
    engine: dict[str, Any],
    judgments: list[dict[str, Any]],
    source: dict[str, Any],
    arms: list[dict[str, Any]],
    guard: SourceGuard,
    started: str,
) -> Path:
    """Assemble the record, write it, and return where it went.

    The record is written even when the guard failed, and the failure is a field
    inside it rather than the absence of a file, so a reader finds the reason
    beside the numbers it disqualifies.
    """

    payload: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "toolkit_version": toolkit_version(),
        "verdict": guard.verdict(len(arms)),
        "run": {
            "name": name,
            "started_at": started,
            "finished_at": now(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "directory": str(run_directory),
        },
        "specification": spec,
        "engine": engine,
        "judgments": judgments,
        "source_project": {**source, "guard": guard.describe()},
        "arms": arms,
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / RECORD_NAME
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_record(run_directory: Path) -> dict[str, Any]:
    """Read a run record, refusing a directory that is not one."""

    path = Path(run_directory).expanduser().resolve() / RECORD_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. A run record is written by "
            "`rag-experiments run`; nothing was recorded there."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    """The timestamp a record's `run` block carries."""

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


__all__ = [
    "RECORD_NAME",
    "RECORD_SCHEMA_VERSION",
    "VERDICT_NO_ARMS",
    "VERDICT_SOURCE_CHANGED",
    "VERDICT_VERIFIED",
    "SourceGuard",
    "close_guard",
    "now",
    "read_record",
    "take_guard",
    "write_record",
]
