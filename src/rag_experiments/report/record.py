"""The record a run leaves, and the verdict that says whether it counts.

Every run writes one JSON file. It is written whether the run succeeded, failed,
or was interrupted, because a failed run with its provenance is the most useful
thing an experiment produces: it names the engine, the corpus digest, the settings,
the exact commands, and the log that holds the output, so the failure is
attributable rather than mysterious.

The guard is part of the record and part of the verdict. A run whose source
project digest moved is a failed measurement, whatever its numbers say, and this
module says so in one field a caller can read without parsing anything else. A run
whose closing digest could not be taken at all is `unknown` rather than unchanged:
the harness promised the corpus was untouched, and it did not get to find out.

`verified` is the narrowest verdict here. It means the source project's bytes were
identical before and after, every arm measured every split it was asked for, the
engine's own files did not move while the arms ran, and nothing stopped the run.
Every other state has its own name, so a reader is never told a run counted when it
did not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import toolkit_version
from ..errors import ExperimentError
from ..sandbox import Snapshot, differences, snapshot

#: The record's own version.
RECORD_SCHEMA_VERSION = 2

#: Where a run's record and its arm directories live inside a runs directory.
RECORD_NAME = "run.json"

#: The verdicts a run can carry. `verified` means the source project's bytes were
#: identical before and after, every arm measured what it was asked to, and the
#: engine did not move underneath the run. Nothing else counts as a measurement.
VERDICT_VERIFIED = "verified"
VERDICT_SOURCE_CHANGED = "source_project_changed"
VERDICT_ENGINE_CHANGED = "engine_source_changed"
VERDICT_INCOMPLETE = "incomplete"
VERDICT_INTERRUPTED = "interrupted"
VERDICT_FAILED = "failed"
VERDICT_NO_ARMS = "no_arm_measured"

#: Every verdict this module can write, so a reader can refuse one it does not
#: know rather than treat it as verified.
VERDICTS = (
    VERDICT_VERIFIED,
    VERDICT_SOURCE_CHANGED,
    VERDICT_ENGINE_CHANGED,
    VERDICT_INCOMPLETE,
    VERDICT_INTERRUPTED,
    VERDICT_FAILED,
    VERDICT_NO_ARMS,
)

#: What the guard could say about the source project.
GUARD_UNCHANGED = "unchanged"
GUARD_CHANGED = "changed"
GUARD_UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SourceGuard:
    """What the source project's bytes were before a run and after it.

    `before` is None when the opening digest could not be taken and `after` is None
    when the closing one could not be, which are both `unknown` rather than
    `unchanged`: the harness promised the corpus was untouched and did not get to
    find out.
    """

    before: Snapshot | None
    after: Snapshot | None
    differences: tuple[str, ...]
    error: str | None = None

    @property
    def state(self) -> str:
        if self.before is None or self.after is None:
            return GUARD_UNKNOWN
        return GUARD_CHANGED if self.differences else GUARD_UNCHANGED

    @property
    def unchanged(self) -> bool:
        return self.state == GUARD_UNCHANGED

    def describe(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "before": None if self.before is None else self.before.describe(),
            "after": None if self.after is None else self.after.describe(),
            "unchanged": self.unchanged,
            "differences": list(self.differences),
            "error": self.error,
            "note": (
                "The digest covers every path in the project except the process "
                "state a serving app rewrites on its own, which is listed under "
                "`excluded_paths`. Any other change while the run was going, made "
                "by this harness, by the app, or by a person, disqualifies the run: "
                "the arms read a corpus that is no longer the one on disk."
            ),
        }


def take_guard(root: Path) -> Snapshot:
    """Digest a source project before a run begins."""

    return snapshot(root)


def close_guard(before: Snapshot | None) -> SourceGuard:
    """Digest the same project again, and say what moved.

    A closing digest that cannot be taken is recorded as unknown rather than
    raising: the run already has arms measured, and losing the record of what they
    measured because the project became unreadable would be the wrong loss. A
    `before` of None means the run never got as far as taking one, which is the
    same unknown for the same reason.
    """

    if before is None:
        return SourceGuard(
            before=None,
            after=None,
            differences=(),
            error="the source project could not be digested before the run began",
        )
    try:
        after = snapshot(before.root)
    except (OSError, ValueError, ExperimentError) as exc:
        # Everything the guard can fail on: the project went away, a path became
        # unreadable, or the app's own names changed under it. Each is recorded as
        # the reason the closing digest could not be taken, because losing the record
        # of what was measured over it would be the worse loss.
        return SourceGuard(before=before, after=None, differences=(), error=str(exc))
    return SourceGuard(
        before=before,
        after=after,
        differences=tuple(differences(before, after)),
    )


def decide_verdict(
    *,
    guard: SourceGuard,
    arms: list[dict[str, Any]],
    engine: dict[str, Any],
    error: str | None,
    interrupted: bool,
    measured_arms: int,
) -> str:
    """The one field a reader checks before believing a number.

    The order is the argument: a corpus that moved says more than anything else,
    then a corpus that could not be checked at all, then an engine that moved
    under the run, then the arms. A run is `verified` only when every one of those
    says it may be.
    """

    if guard.state == GUARD_CHANGED:
        return VERDICT_SOURCE_CHANGED
    if guard.state == GUARD_UNKNOWN:
        return VERDICT_FAILED
    if engine.get("changed"):
        return VERDICT_ENGINE_CHANGED
    if engine.get("content_sha256_after") is None:
        # The engine's own files could not be digested after the arms ran, so this
        # run cannot say the code stayed still while it measured.
        return VERDICT_FAILED
    if interrupted:
        return VERDICT_INTERRUPTED
    if not measured_arms:
        return VERDICT_FAILED if error is not None else VERDICT_NO_ARMS
    if error is not None or measured_arms != len(arms):
        return VERDICT_INCOMPLETE
    return VERDICT_VERIFIED


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
    verdict: str | None = None,
    error: str | None = None,
    interruption: str | None = None,
    measured_arms: int | None = None,
) -> Path:
    """Assemble the record, write it, and return where it went.

    The record is written even when the guard failed, the engine moved, or the run
    was interrupted, and the failure is a field inside it rather than the absence
    of a file, so a reader finds the reason beside the numbers it disqualifies.
    `verdict` is computed here when a caller does not state one, so no caller can
    write a record whose verdict disagrees with its own evidence.
    """

    counted = (
        len([arm for arm in arms if arm.get("measured")])
        if measured_arms is None
        else measured_arms
    )
    if verdict is None:
        verdict = decide_verdict(
            guard=guard,
            arms=arms,
            engine=engine,
            error=error,
            interrupted=interruption is not None,
            measured_arms=counted,
        )
    payload: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "toolkit_version": toolkit_version(),
        "verdict": verdict,
        "run": {
            "name": name,
            "started_at": started,
            "finished_at": now(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "directory": str(run_directory),
            "complete": error is None and interruption is None,
        },
        "error": error,
        "interruption": interruption,
        "measured_arm_count": counted,
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
    "GUARD_CHANGED",
    "GUARD_UNCHANGED",
    "GUARD_UNKNOWN",
    "RECORD_NAME",
    "RECORD_SCHEMA_VERSION",
    "VERDICTS",
    "VERDICT_ENGINE_CHANGED",
    "VERDICT_FAILED",
    "VERDICT_INCOMPLETE",
    "VERDICT_INTERRUPTED",
    "VERDICT_NO_ARMS",
    "VERDICT_SOURCE_CHANGED",
    "VERDICT_VERIFIED",
    "SourceGuard",
    "close_guard",
    "decide_verdict",
    "now",
    "read_record",
    "take_guard",
    "write_record",
]
