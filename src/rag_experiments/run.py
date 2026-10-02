"""One run, from a specification to a record, with the guard around it.

The order here is the whole point of the toolkit and is not to be rearranged:

1. Digest the source project. Nothing has been touched yet, so this is what
   "untouched" will be measured against.
2. Measure each arm in its own sandbox.
3. Digest the source project again, whatever happened in between.
4. Write the record, then report the verdict.

A run whose digest moved is a failed measurement even when every arm exited
cleanly, so the verdict is computed after the arms rather than alongside them.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine.locate import EngineSource
from .errors import ExperimentError
from .experiment import ArmResult, RunSpec, run_arm
from .report import record as run_record
from .report.compare import render_comparison


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Where a run left its record, and what the guard said about the source."""

    record_path: Path
    record: dict[str, Any]
    comparison: str
    results: tuple[ArmResult, ...]

    @property
    def verdict(self) -> str:
        return str(self.record.get("verdict") or "")

    @property
    def verified(self) -> bool:
        return self.verdict == run_record.VERDICT_VERIFIED


def run_experiment(
    spec: RunSpec,
    *,
    app_source: EngineSource,
    workspace: Path,
    runs_directory: Path,
    keep_sandboxes: bool = True,
    validate_first: bool = True,
) -> RunOutcome:
    """Measure every arm of a specification and write one record for the run."""

    if not spec.source_project.is_dir():
        raise ExperimentError(
            f"The specification's source project is not there: {spec.source_project}"
        )

    workspace = Path(workspace).expanduser().resolve()
    run_directory = (Path(runs_directory).expanduser().resolve()) / _run_id(spec.name)
    if run_directory.exists():
        raise ExperimentError(
            f"A run directory already exists at {run_directory}. A run never "
            "overwrites a record; remove the directory or rename the "
            "specification's `name`."
        )
    run_directory.mkdir(parents=True)

    started = run_record.now()
    started_at = time.perf_counter()
    guard_before = run_record.take_guard(spec.source_project)

    results: list[ArmResult] = []
    for arm in spec.arms:
        result = run_arm(
            spec,
            arm,
            workspace=workspace,
            run_directory=run_directory,
            app_source=app_source,
            keep_sandbox=keep_sandboxes,
            validate_first=validate_first,
        )
        results.append(result)

    guard = run_record.close_guard(guard_before)
    arms = [result.describe() for result in results]
    record_path = run_record.write_record(
        run_directory,
        name=spec.name,
        elapsed_seconds=time.perf_counter() - started_at,
        spec=spec.describe(),
        engine=app_source.describe(),
        judgments=_judgments(spec),
        source={
            "project_root": str(spec.source_project),
            "pointed_generation": _pointed_generation(spec),
        },
        arms=arms,
        guard=guard,
        started=started,
    )
    record = run_record.read_record(run_directory)
    return RunOutcome(
        record_path=record_path,
        record=record,
        comparison=render_comparison(record),
        results=tuple(results),
    )


def _judgments(spec: RunSpec) -> list[dict[str, Any]]:
    """Each judged split as the record states it: where it was and what it held.

    The digest is over the bytes, so a run against a judged set that was edited
    afterwards is distinguishable from one against the same set as it was. Two
    splits measured in one run are listed together, because a development number
    and a held-out number are only comparable when the same arms produced both.
    """

    return [
        {
            "name": split.name,
            "path": str(split.path),
            "sha256": _digest(split.path),
            "byte_count": split.path.stat().st_size,
        }
        for split in spec.judgments
    ]


def _pointed_generation(spec: RunSpec) -> str | None:
    from .sandbox import read_layout, selected_generation

    try:
        return selected_generation(read_layout(spec.source_project))
    except ExperimentError:
        return None


def _digest(path: Path) -> str:
    accumulator = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            accumulator.update(block)
    return accumulator.hexdigest()


def _run_id(name: str) -> str:
    """A run's directory name: its specification's name, and a timestamp.

    A run never overwrites another, so the identifier carries when it happened as
    well as what it was called.
    """

    stamp = run_record.now().replace(":", "").replace("-", "").rstrip("Z")
    safe = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in name
    ).strip("-")
    return f"{safe or 'run'}-{stamp}"


__all__ = ["RunOutcome", "run_experiment"]
