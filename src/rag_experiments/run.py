"""One run, from a specification to a record, with the guard around it.

The order here is the whole point of the toolkit and is not to be rearranged:

1. Digest the source project. Nothing has been touched yet, so this is what
   "untouched" will be measured against.
2. Measure each arm in its own sandbox, writing every byte it produces inside this
   run's own directory.
3. Digest the source project again, whatever happened in between, and digest the
   engine's own files so a code change during the run cannot hide inside a number.
4. Write the record, then report the verdict.

The record is written on every exit. A run whose first arm refuses, whose settings
are invalid, whose report is malformed, or whose terminal was closed leaves a
record naming what it did, which arm stopped it, the log that holds the output, and
what the guard said about the source. The guard's own failure is recorded as
`unknown` rather than raised, because losing the record of what was measured
because the project became unreadable would be the wrong loss.

A run whose digest moved is a failed measurement even when every arm exited
cleanly, so the verdict is computed after the arms rather than alongside them.

This isolation is cooperative. Every child this harness spawns is given a sandbox,
a pinned settings file, an environment that names the tree under test, and paths
that point inside the copy. Nothing here is an operating-system sandbox: code the
run measures, including a code arm's patch, runs with the reader's own privileges
and could reach the original if it tried. What this harness proves is that the
harness itself never wrote to the source, which is a claim about this code and not
about the code under test.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine.locate import EngineSource, tree_digest
from .errors import ExperimentError
from .experiment import ArmFailure, ArmResult, RunSpec, run_arm, segment
from .experiment.execute import ARM_EVIDENCE
from .report import record as run_record
from .report.compare import render_comparison
from .sandbox import Snapshot, read_layout, refuse_nested, selected_generation

#: Where a run keeps the judged sets it measured from. The bytes are copied rather
#: than referenced, because a judged set edited after the run would leave a record
#: whose digest and whose content disagree.
JUDGED_DIRECTORY = "judged"


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Where a run left its record, and what the guard said about the source."""

    record_path: Path
    record: dict[str, Any]
    comparison: str
    results: tuple[ArmResult, ...]
    run_id: str
    run_directory: Path

    @property
    def verdict(self) -> str:
        return str(self.record.get("verdict") or "")

    @property
    def verified(self) -> bool:
        return self.verdict == run_record.VERDICT_VERIFIED

    @property
    def error(self) -> str | None:
        value = self.record.get("error")
        return None if value is None else str(value)

    @property
    def complete(self) -> bool:
        return not self.error and not self.record.get("interruption")

    @property
    def guard_state(self) -> str:
        guard = (self.record.get("source_project") or {}).get("guard") or {}
        return str(guard.get("state") or "unknown")


def run_experiment(
    spec: RunSpec,
    *,
    app_source: EngineSource,
    workspace: Path,
    runs_directory: Path,
    keep_sandboxes: bool = True,
    validate_first: bool = True,
) -> RunOutcome:
    """Measure every arm of a specification and write one record for the run.

    Nothing escapes without a record: a refusal, an interruption, and a run that
    never started an arm all leave one, and the caller learns the verdict from the
    returned outcome rather than from an exception that carried no numbers.
    """

    _refuse_before_anything(spec, app_source, workspace, runs_directory)

    workspace = Path(workspace).expanduser().resolve()
    runs = Path(runs_directory).expanduser().resolve()
    runs.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(spec.name, runs)
    run_directory = runs / run_id
    run_directory.mkdir()

    started = run_record.now()
    started_at = time.perf_counter()
    guard_before: Snapshot | None = None
    engine_before: dict[str, Any] | None = None

    results: list[ArmResult] = []
    error: str | None = None
    interruption: str | None = None
    finalized: RunOutcome | None = None

    try:
        guard_before = run_record.take_guard(spec.source_project)
        engine_before = app_source.describe()
        # The arms fill the caller's own list, so an interrupt part-way through a
        # run still has the arms that finished, and the arm that was interrupted,
        # when the record is written below.
        error = _measure_all(
            spec,
            results=results,
            app_source=app_source,
            workspace=workspace,
            run_directory=run_directory,
            run_id=run_id,
            keep_sandboxes=keep_sandboxes,
            validate_first=validate_first,
        )
    except BaseException as exc:
        # A KeyboardInterrupt or a SystemExit lands here, and so does anything the
        # arms did not already turn into a refusal. The record is written by the
        # `finally` below, and this handler only says what interrupted the run.
        interruption = (
            f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        )
        raise
    finally:
        finalized = _finalize(
            spec,
            run_directory=run_directory,
            run_id=run_id,
            started=started,
            started_at=started_at,
            guard_before=guard_before,
            engine_before=engine_before,
            app_source=app_source,
            results=results,
            error=error,
            interruption=interruption,
        )
        if interruption is not None:
            print(
                f"rag-experiments: the run was interrupted and its record is at "
                f"{finalized.record_path}",
                file=sys.stderr,
            )

    return finalized


def new_run_id(name: str, runs_directory: Path) -> str:
    """A run directory name that has never been used: the specification's, and a stamp.

    The stamp carries microseconds and the loop refuses an identifier already on
    disk, so two runs of the same specification in the same millisecond are still
    two runs rather than one run overwriting the other's evidence.
    """

    safe = segment(name, strict=False) or "run"
    while True:
        stamp = run_record.now().replace(":", "").replace("-", "").rstrip("Z")
        candidate = f"{safe}-{stamp}"
        if not (Path(runs_directory) / candidate).exists():
            return candidate


def _measure_all(
    spec: RunSpec,
    *,
    results: list[ArmResult],
    app_source: EngineSource,
    workspace: Path,
    run_directory: Path,
    run_id: str,
    keep_sandboxes: bool,
    validate_first: bool,
) -> str | None:
    """Measure each arm in turn, stopping at the first refusal, and say why.

    One arm refusing stops the run rather than continuing, because the arms of a
    specification are a comparison and a later arm measured against a corpus an
    earlier arm disturbed says nothing. The results list is the caller's, so an
    interrupt on the way out still leaves it holding every arm that was attempted.

    The message returned is the one the record carries as its error.
    """

    for arm in spec.arms:
        try:
            results.append(
                run_arm(
                    spec,
                    arm,
                    workspace=workspace,
                    sandbox_name=f"{run_id}-{segment(arm.name)}",
                    arm_directory=run_directory / segment(arm.name),
                    app_source=app_source,
                    keep_sandbox=keep_sandboxes,
                    validate_first=validate_first,
                )
            )
        except ArmFailure as failure:
            results.append(failure.result)
            return str(failure)
        except ExperimentError as exc:
            # An arm's own refusal is already an `ArmFailure`; anything else here
            # is this harness failing to run one, and it stops the run the same way
            # with the arms that had finished kept beside it.
            return str(exc)
        except BaseException as exc:
            # An interrupt is not a refusal and is not swallowed here, but the arm
            # it interrupted carries what it had done, and that belongs in the
            # record the run writes on its way out.
            partial = getattr(exc, ARM_EVIDENCE, None)
            if isinstance(partial, ArmResult):
                results.append(partial)
            raise
    return None


def _finalize(
    spec: RunSpec,
    *,
    run_directory: Path,
    run_id: str,
    started: str,
    started_at: float,
    guard_before: Snapshot | None,
    engine_before: dict[str, Any] | None,
    app_source: EngineSource,
    results: list[ArmResult],
    error: str | None,
    interruption: str | None,
) -> RunOutcome:
    """Write the record, whatever happened, and never raise doing it."""

    guard = run_record.close_guard(guard_before)
    engine_after, engine_error = _engine_digest(app_source)
    before_digest = (engine_before or {}).get("content_sha256")
    engine = {
        **(engine_before or {}),
        "content_sha256_after": engine_after,
        "content_sha256_error": engine_error,
        "changed": (
            None
            if engine_after is None or before_digest is None
            else engine_after != before_digest
        ),
        "note": (
            "A digest of the engine's own src, scripts, and packaged metadata, taken "
            "again after the arms ran. A change here means the run measured two "
            "engines and reports one number."
        ),
    }
    arms = [result.describe() for result in results]
    measured = sum(1 for result in results if result.measured)
    try:
        record_path = run_record.write_record(
            run_directory,
            name=spec.name,
            elapsed_seconds=time.perf_counter() - started_at,
            spec={**spec.describe(), "run_id": run_id},
            engine=engine,
            judgments=_judgments(spec, run_directory),
            source={
                "project_root": str(spec.source_project),
                "pointed_generation": _pointed_generation(spec),
            },
            arms=arms,
            guard=guard,
            started=started,
            error=error,
            interruption=interruption,
            measured_arms=measured,
        )
        record = run_record.read_record(run_directory)
    except OSError as exc:
        # A record that cannot be written is the one failure this run cannot report
        # about itself, so it is raised with the directory a reader would look in.
        raise ExperimentError(
            f"The run's record could not be written to {run_directory}: {exc}"
        ) from exc
    return RunOutcome(
        record_path=record_path,
        record=record,
        comparison=_comparison(record),
        results=tuple(results),
        run_id=run_id,
        run_directory=run_directory,
    )


def _engine_digest(app_source: EngineSource) -> tuple[str | None, str | None]:
    """Digest the engine tree after the arms ran, or say why it could not be done.

    A digest that cannot be taken is None with its reason rather than an omission,
    and a run whose engine could not be read afterwards is never verified: this run
    cannot say the code stayed still while it measured.
    """

    try:
        return tree_digest(app_source.root), None
    except (OSError, ValueError, ExperimentError) as exc:
        return None, str(exc)


def _comparison(record: dict[str, Any]) -> str:
    """The table for a record, or the reason it has none yet.

    The table is a reading of the record and not part of it, so a report this
    record's version does not match is a message rather than a refusal of a run
    that has already been written.
    """

    try:
        return render_comparison(record)
    except (ExperimentError, OSError, ValueError) as exc:
        return f"the comparison table was not rendered: {exc}"


def _refuse_before_anything(
    spec: RunSpec,
    app_source: EngineSource,
    workspace: Path,
    runs_directory: Path,
) -> None:
    """Refuse a placement that would make the guard watch the run's own files.

    A workspace or a runs directory inside the source project would have every
    sandbox and every report written under the tree the guard digests, so the run
    would report itself as the thing that changed the corpus. One that contains the
    project would let a cleanup remove it.
    """

    if not spec.source_project.is_dir():
        raise ExperimentError(
            f"The specification's source project is not there: {spec.source_project}"
        )
    for area, what in (
        (Path(workspace), "The workspace"),
        (Path(runs_directory), "The runs directory"),
    ):
        refuse_nested(area, spec.source_project, what=what, inside="source project")
        refuse_nested(area, app_source.root, what=what, inside="engine tree")


def _judgments(spec: RunSpec, run_directory: Path) -> list[dict[str, Any]]:
    """Each judged split as the record states it: where it was, and what it held.

    The bytes are copied under the run directory, because a judged set edited after
    the run would otherwise leave a record whose digest describes a file no reader
    can find again. Two splits measured in one run are listed together, because a
    development number and a held-out number are only comparable when the same arms
    produced both.
    """

    kept = run_directory / JUDGED_DIRECTORY
    entries: list[dict[str, Any]] = []
    for split in spec.judgments:
        entry: dict[str, Any] = {
            "name": split.name,
            "path": str(split.path),
            "sha256": _digest(split.path),
            "byte_count": split.path.stat().st_size,
        }
        try:
            kept.mkdir(parents=True, exist_ok=True)
            copy = kept / f"{segment(split.name)}.json"
            shutil.copyfile(split.path, copy)
            entry["kept_at"] = str(copy)
            entry["kept_sha256"] = _digest(copy)
        except OSError as exc:
            entry["kept_error"] = str(exc)
        entries.append(entry)
    return entries


def _pointed_generation(spec: RunSpec) -> str | None:
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


__all__ = ["RunOutcome", "new_run_id", "run_experiment"]
