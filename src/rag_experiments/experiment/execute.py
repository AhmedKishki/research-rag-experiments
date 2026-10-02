"""Run one arm, and return what it produced.

An arm is measured by running the app's own evaluation harness as a subprocess,
against a sandbox, through an engine tree. Nothing is computed here: the harness
writes the report, the harness writes the numbers, and this module only says what
it ran, what it returned, whether the report is the measurement that was asked
for, and how long it took.

An arm fails loudly and keeps what it had. A non-zero exit, a missing report, and
a report that does not match the run's own arguments are all refusals rather than
a row of zeroes, and every one of them raises `ArmFailure` carrying the evidence
gathered so far, so a run that stops still writes a record naming the command that
stopped it, the log that holds its output, and the arms that had already finished.

The arm's evidence lives in the run's own directory: the log, each split's report,
the report's digest, and a code arm's checkout. A sandbox is a large disposable
copy in the workspace and may be removed; an evidence file is small and is never
removed by this module.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..engine.harness import HarnessContract, harness_contract
from ..engine.locate import EngineSource
from ..engine.runner import (
    describe_child_environment,
    resolve_settings_for_tree,
)
from ..errors import ExperimentError
from ..sandbox import Sandbox, pin_settings, refuse_nested
from ..sandbox import create as create_sandbox
from .checkout import Checkout, make_checkout
from .report_schema import (
    SELECTION_FLAGS,
    QuerySelection,
    ReportFacts,
    ReportValidationError,
    select_queries,
    validate_report,
)
from .spec import (
    HARNESS_JOINED,
    HARNESS_REPEATED,
    HARNESS_SETTINGS,
    Arm,
    JudgedSet,
    RunSpec,
    segment,
)

#: How long a prepare step or a measurement is given before it is called hung. A
#: full ingestion of a large corpus on one CPU is an hour, and a query sweep over
#: thirty queries with a cross-encoder is minutes, so both are bounded well above
#: what they need.
PREPARE_TIMEOUT_SECONDS = 4 * 60 * 60
MEASURE_TIMEOUT_SECONDS = 4 * 60 * 60

#: The harness arguments this module always supplies, so a specification cannot
#: redirect the corpus, the judged set, or where the report lands.
JUDGMENTS_FLAG = "--judgments"
REPORT_FLAG = "--report"
PROJECT_FLAG = "--project"
VALIDATE_FLAG = "--validate-only"

#: How far a split's report sits from the arm's directory. A split name is a path
#: segment, which `spec.segment` refuses to let be anything else.
REPORT_PATTERN = "report-{split}.json"

#: Where a code arm's checkout goes inside its own evidence directory.
ENGINE_DIRECTORY = "engine"

#: The attribute an interrupt carries out of an arm, holding what the arm had done
#: when the interrupt arrived. It is named rather than assumed so the run that
#: catches it reads its own arm evidence and nothing else.
ARM_EVIDENCE = "rag_experiments_arm_result"


class ArmFailure(ExperimentError):
    """One arm refused, carrying whatever it had produced when it did.

    The exception is the message a reader is shown; the result beside it is what
    the run's record is written from, so a failure keeps the commands it ran, the
    reports it wrote, and the log that holds the output.
    """

    def __init__(self, message: str, *, result: ArmResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True, slots=True)
class Measurement:
    """One harness invocation: a stage, a split, a command, and what it returned."""

    stage: str
    split: str
    command: list[str]
    exit_code: int
    elapsed_seconds: float
    stdout: str
    stderr: str
    report: Path | None = None
    report_facts: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        record = {
            "stage": self.stage,
            "split": self.split,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }
        if self.report is not None:
            record["report"] = str(self.report)
            record["report_sha256"] = (
                self.report_facts.get("sha256") if (self.report_facts) else None
            )
        if self.report_facts is not None:
            record["report_facts"] = self.report_facts
        return record


@dataclass(frozen=True, slots=True)
class ArmResult:
    """What one arm did, and where its evidence is."""

    arm: Arm
    engine: EngineSource | None
    checkout: Checkout | None
    sandbox: Sandbox | None
    settings: dict[str, Any]
    environment: dict[str, Any]
    prepare: list[dict[str, Any]]
    validate: tuple[Measurement, ...]
    measure: tuple[Measurement, ...]
    log_path: Path
    measured_splits: tuple[str, ...] = ()
    failure: str | None = None
    failure_stage: str | None = None
    sandbox_removed: bool = False

    @property
    def measured(self) -> bool:
        """Whether every split this arm was asked for produced a valid report."""

        return (
            not self.failure
            and bool(self.measure)
            and not any(
                item.exit_code != 0 or item.report is None for item in self.measure
            )
        )

    def reports(self) -> dict[str, Path]:
        """Where each split's report was written, keyed by split name."""

        return {
            item.split: item.report for item in self.measure if item.report is not None
        }

    def describe(self) -> dict[str, Any]:
        return {
            "arm": self.arm.describe(),
            "engine": None if self.engine is None else self.engine.describe(),
            "checkout": None if self.checkout is None else self.checkout.describe(),
            "sandbox": None if self.sandbox is None else self.sandbox.describe(),
            "settings": self.settings,
            "environment": self.environment,
            "prepare": self.prepare,
            "validate": [item.describe() for item in self.validate],
            "measure": [item.describe() for item in self.measure],
            "reports": {name: str(path) for name, path in self.reports().items()},
            "log": str(self.log_path),
            "measured_splits": list(self.measured_splits),
            "measured": self.measured,
            "failure": self.failure,
            "failure_stage": self.failure_stage,
            "sandbox_removed": self.sandbox_removed,
        }


@dataclass(frozen=True, slots=True)
class PreparedArm:
    """An arm that exists on disk: its sandbox, its engine, and its pinned settings.

    This is what a dry run stops at and what a real run measures next. The two
    share it, so a dry run answers the question a real run will actually hit
    rather than a question one layer above it.
    """

    arm: Arm
    engine: EngineSource
    checkout: Checkout | None
    sandbox: Sandbox
    settings: dict[str, Any]
    environment: dict[str, Any]
    log_path: Path
    report_path: Path
    contract: HarnessContract
    selections: dict[str, QuerySelection]

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.engine.describe(),
            "checkout": None if self.checkout is None else self.checkout.describe(),
            "sandbox": self.sandbox.describe(),
            "settings": self.settings,
            "environment": self.environment,
            "log": str(self.log_path),
            "query_selection": {
                name: selection.describe()
                for name, selection in self.selections.items()
            },
        }


@dataclass(slots=True)
class _Progress:
    """What an arm has done so far, so a refusal still has something to report."""

    engine: EngineSource | None = None
    checkout: Checkout | None = None
    sandbox: Sandbox | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    prepare: list[dict[str, Any]] = field(default_factory=list)
    validate: list[Measurement] = field(default_factory=list)
    measure: list[Measurement] = field(default_factory=list)
    measured_splits: list[str] = field(default_factory=list)
    failure: str | None = None
    failure_stage: str | None = None
    sandbox_removed: bool = False

    def result(self, arm: Arm, log_path: Path) -> ArmResult:
        return ArmResult(
            arm=arm,
            engine=self.engine,
            checkout=self.checkout,
            sandbox=self.sandbox,
            settings=self.settings,
            environment=self.environment,
            prepare=list(self.prepare),
            validate=tuple(self.validate),
            measure=tuple(self.measure),
            log_path=log_path,
            measured_splits=tuple(self.measured_splits),
            failure=self.failure,
            failure_stage=self.failure_stage,
            sandbox_removed=self.sandbox_removed,
        )


def prepare_arm(
    spec: RunSpec,
    arm: Arm,
    *,
    workspace: Path,
    sandbox_name: str,
    arm_directory: Path,
    app_source: EngineSource,
) -> PreparedArm:
    """Materialize an arm's sandbox, its engine, and its pinned settings.

    Each arm gets its own sandbox, in its own run's workspace, and every file it
    writes is inside that run's directory. A dry run and the real run after it are
    two runs with two run identifiers, so the second never collides with the first
    and neither overwrites the other's evidence.
    """

    arm_directory.mkdir(parents=True, exist_ok=True)
    log_path = arm_directory / "arm.log"
    progress = _Progress()
    log = _Log(log_path)
    log.emit(f"arm: {arm.name} ({arm.kind})")
    log.emit(f"run directory: {arm_directory}")

    refuse_nested(
        Path(workspace).expanduser().resolve() / sandbox_name,
        spec.source_project,
        what="The arm's sandbox directory",
        inside="source project",
    )
    refuse_nested(
        arm_directory,
        spec.source_project,
        what="The arm's evidence directory",
        inside="source project",
    )
    refuse_nested(
        Path(workspace).expanduser().resolve() / sandbox_name,
        app_source.root,
        what="The arm's sandbox directory",
        inside="engine tree",
    )
    refuse_nested(
        arm_directory,
        app_source.root,
        what="The arm's evidence directory",
        inside="engine tree",
    )

    try:
        sandbox = create_sandbox(
            workspace,
            name=sandbox_name,
            source=spec.source_project,
            generations=spec.generations,
        )
        progress.sandbox = sandbox
        log.emit(f"sandbox: {sandbox.root}")
        log.emit(f"  generations: {', '.join(sandbox.generation_ids)}")

        checkout: Checkout | None = None
        engine = app_source
        progress.engine = app_source
        if arm.kind == "code":
            tree_root = arm_directory / ENGINE_DIRECTORY
            log.emit(f"checkout: {app_source.root} -> {tree_root}")
            checkout = make_checkout(
                tree_root,
                app_source=app_source,
                base=arm.base,
                patch=arm.patch,
                evidence_directory=arm_directory,
            )
            progress.checkout = checkout
            log.emit(
                f"checkout base {checkout.base_revision}, method {checkout.method}, "
                f"untracked files carried {len(checkout.source_untracked)}"
            )
            engine = locate_or_refuse(tree_root)
            progress.engine = engine

        overlay = effective_overlay(arm, spec.harness)
        log.emit(f"resolving settings in {engine.root} with overlay {overlay or '{}'}")
        answer = resolve_settings_for_tree(
            engine,
            project=spec.source_project,
            overlay=overlay,
            pin_project=sandbox.root,
        )
        pin_settings(sandbox, answer.document)
        settings = answer.describe()
        settings["written_to"] = str(sandbox.settings_file)
        settings["config_home"] = str(sandbox.config_home)
        settings["model_cache_root"] = answer.model_cache_root
        progress.settings = settings
        log.emit(
            f"pinned {len(answer.values)} settings to {sandbox.settings_file} "
            f"({len(overlay)} overridden); model cache "
            f"{answer.model_cache_root or 'unpinned'}"
        )
        environment = describe_child_environment(
            engine, config_home=sandbox.config_home
        )
        progress.environment = environment
        log.emit(
            f"child environment: dropped {len(environment['dropped_variables'])} "
            f"settings variables, set {', '.join(sorted(environment['given']))}"
        )
        # The query set every report will be checked against, computed from the judged
        # file and the run's own selection flags. A specification that selects a class
        # the file does not hold is refused here rather than after the arms have spent
        # an hour measuring a report that could never be believed.
        selections = {split.name: _selection(spec, split) for split in spec.judgments}
        for name, selection in selections.items():
            log.emit(
                f"selected {len(selection.query_ids)} of "
                f"{selection.total_query_count} queries for {name}: "
                f"{', '.join(selection.query_ids)}"
            )
        return PreparedArm(
            arm=arm,
            engine=engine,
            checkout=checkout,
            sandbox=sandbox,
            settings=settings,
            environment=environment,
            log_path=log_path,
            report_path=arm_directory / REPORT_PATTERN.format(split="single"),
            contract=harness_contract(engine),
            selections=selections,
        )
    except ExperimentError as exc:
        # A refusal before the harness ran is still the arm's outcome: the record
        # says which arm stopped the run and why, so it cannot read as an arm that
        # simply measured nothing.
        progress.failure = str(exc)
        progress.failure_stage = "prepare"
        log.emit(f"REFUSED at prepare: {str(exc).splitlines()[0]}")
        raise ArmFailure(str(exc), result=progress.result(arm, log_path)) from exc


def run_arm(
    spec: RunSpec,
    arm: Arm,
    *,
    workspace: Path,
    sandbox_name: str,
    arm_directory: Path,
    app_source: EngineSource,
    keep_sandbox: bool = True,
    validate_first: bool = True,
) -> ArmResult:
    """Prepare one arm, optionally prepare the corpus, measure it, and report.

    `validate_first` runs the app's harness in its validation stage before any
    search, so a judged target that cannot be resolved stops the run before it
    costs an hour of inference rather than after.

    Every refusal is raised as `ArmFailure` with the evidence gathered so far, and
    the arm's log is written as it happens rather than at the end, so an
    interruption leaves the output that explains it.
    """

    progress = _Progress()
    log = _Log(arm_directory / "arm.log")
    try:
        prepared = prepare_arm(
            spec,
            arm,
            workspace=workspace,
            sandbox_name=sandbox_name,
            arm_directory=arm_directory,
            app_source=app_source,
        )
    except ArmFailure:
        # Raised rather than returned: a refused arm stops the run, and a run that
        # stopped has a failure to report rather than an arm that measured nothing.
        raise
    except ExperimentError as exc:
        progress.failure = str(exc)
        progress.failure_stage = "prepare"
        log.emit(f"REFUSED at prepare: {str(exc).splitlines()[0]}")
        raise ArmFailure(str(exc), result=progress.result(arm, log.path)) from exc

    sandbox = prepared.sandbox
    engine = prepared.engine
    progress.engine = engine
    progress.checkout = prepared.checkout
    progress.sandbox = sandbox
    progress.settings = prepared.settings
    progress.environment = prepared.environment

    def refuse(stage: str, message: str) -> ArmFailure:
        progress.failure = message
        progress.failure_stage = stage
        log.emit(f"REFUSED at {stage}: {message.splitlines()[0]}")
        return ArmFailure(message, result=progress.result(arm, log.path))

    environment = _child_environment(engine, sandbox)
    try:
        for command in arm.prepare:
            arguments = _interpolate(command, sandbox, engine)
            log.emit(f"prepare: {arguments}")
            outcome = _run(
                arguments,
                environment,
                cwd=sandbox.root,
                timeout=PREPARE_TIMEOUT_SECONDS,
            )
            outcome["command"] = arguments
            progress.prepare.append(outcome)
            log.outcome(outcome)
            if outcome["exit_code"] != 0:
                raise refuse(
                    "prepare",
                    f"Arm {arm.name!r} failed at the prepare step with exit "
                    f"{outcome['exit_code']}.\n"
                    f"  command: {' '.join(arguments)}\n"
                    f"  output:  {_tail(outcome.get('stderr') or outcome.get('stdout'))}\n"
                    f"  full:    {log.path}",
                )

        # The app's own harness resolves a judged target before it searches, and a
        # target it cannot resolve uniquely makes every number for that split
        # meaningless. That check runs first because it costs no inference and finds
        # the failure in seconds rather than after an hour of cross-encoding.
        if validate_first:
            for split in spec.judgments:
                item = _invoke(
                    spec,
                    engine,
                    sandbox,
                    prepared,
                    environment,
                    stage="validate",
                    split=split,
                    log=log,
                )
                progress.validate.append(item)
                log.emit(
                    f"validate[{split.name}] exit {item.exit_code} in "
                    f"{item.elapsed_seconds:.1f} s"
                )
                log.output(item)
                if item.exit_code != 0:
                    raise refuse(
                        f"validate ({split.name})",
                        f"Arm {arm.name!r} stopped in the validation pass for split "
                        f"{split.name!r} with exit {item.exit_code}.\n"
                        f"  because:  The judged set {split.path} could not be "
                        f"resolved against this arm's generation. Every number for "
                        f"this split would describe nothing, so the run stops here "
                        f"rather than after measuring it.\n"
                        f"  command: {' '.join(item.command)}\n"
                        f"  output:  {_tail(item.stderr or item.stdout)}\n"
                        f"  full:    {log.path}",
                    )

        for split in spec.judgments:
            item = _invoke(
                spec,
                engine,
                sandbox,
                prepared,
                environment,
                stage="measure",
                split=split,
                log=log,
            )
            progress.measure.append(item)
            log.emit(
                f"measure[{split.name}] exit {item.exit_code} in "
                f"{item.elapsed_seconds:.1f} s"
            )
            log.output(item)
            if item.exit_code != 0:
                raise refuse(
                    f"measure ({split.name})",
                    f"Arm {arm.name!r} failed measuring split {split.name!r} with "
                    f"exit {item.exit_code}.\n"
                    f"  command: {' '.join(item.command)}\n"
                    f"  output:  {_tail(item.stderr or item.stdout)}\n"
                    f"  full:    {log.path}",
                )
            facts = _require_report(
                arm,
                split,
                item,
                log,
                prepared.contract,
                spec,
                prepared.selections[split.name],
            )
            log.emit(
                f"report[{split.name}] schema {facts.schema_version}, modes "
                f"{', '.join(facts.modes)}, {facts.run_count} runs, "
                f"sha256 {facts.sha256[:16]}"
            )
            progress.measured_splits.append(split.name)
            progress.measure[-1] = _with_facts(item, facts)
    except ArmFailure:
        raise
    except ReportValidationError as exc:
        raise refuse("report", str(exc)) from exc
    except ExperimentError as exc:
        raise refuse("run", str(exc)) from exc
    except BaseException as exc:
        # A KeyboardInterrupt or a SystemExit is not a refusal, so it continues
        # outward rather than being wrapped. What this arm had already done is
        # attached to the exception on its way, so the record the run writes in its
        # `finally` still carries the arm's log, its reports, and its commands.
        progress.failure = (
            f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        )
        progress.failure_stage = "interrupted"
        log.emit(f"INTERRUPTED: {progress.failure}")
        setattr(exc, ARM_EVIDENCE, progress.result(arm, log.path))
        raise

    if not keep_sandbox:
        from ..sandbox import remove as remove_sandbox

        remove_sandbox(Path(workspace), sandbox.name)
        progress.sandbox_removed = True
        log.emit(f"removed sandbox {sandbox.root}; this arm's evidence stays put")
    return progress.result(arm, log.path)


def _require_report(
    arm: Arm,
    split: JudgedSet,
    item: Measurement,
    log: _Log,
    contract: HarnessContract,
    spec: RunSpec,
    selection: QuerySelection,
) -> ReportFacts:
    """Read the report and refuse it unless it is this arm's measurement."""

    if item.report is None or not item.report.is_file():
        raise ReportValidationError(
            f"Arm {arm.name!r} exited cleanly and wrote no report for split "
            f"{item.split!r} at {item.report}. The full output is in {log.path}."
        )
    return validate_report(
        item.report,
        contract=contract,
        requested_modes=requested_modes(spec, contract),
        selection=selection,
        judged_sha256=_digest_of(split.path),
        top_k=_as_int(spec.harness.get("top_k")),
        deep_top_k=_as_int(spec.harness.get("deep_top_k")),
        arm=arm.name,
        split=split.name,
    )


def _selection(spec: RunSpec, split: JudgedSet) -> QuerySelection:
    """The queries this run measures, from the judged file and the run's own flags.

    The flags are the ones the harness was given on the command line, and the file is
    the one it was handed, so the expected query set is the engine's set rather than
    a second implementation of a similar one. A judged set that selects nothing, or
    that repeats a query id, is refused before any arm measures it.
    """

    return select_queries(
        split.path,
        **{
            flag: spec.harness.get(flag)
            for flag in SELECTION_FLAGS
            if spec.harness.get(flag) is not None
        },
    )


def requested_modes(
    spec: RunSpec, contract: HarnessContract | None = None
) -> tuple[str, ...]:
    """The modes this run asked the harness to measure.

    A specification that names `modes` states them. One that does not is asking for
    everything the engine's harness measures, which is read from the harness itself
    through `contract`; without a contract to ask, this refuses rather than
    inventing a mode list, because a mode this harness guessed would be refused by
    the engine's own parser or, worse, silently absent from a report.
    """

    named = spec.harness.get("modes")
    if isinstance(named, str) and named.strip():
        return tuple(item.strip() for item in named.split(",") if item.strip())
    if isinstance(named, (list, tuple)) and named:
        return tuple(str(mode) for mode in named)
    if contract is None:
        raise ExperimentError(
            "The specification names no `modes`, and no engine contract was read, so "
            "the modes this run asked for cannot be stated. Name `modes` in the "
            "specification's `harness` block."
        )
    return contract.modes


def _invoke(
    spec: RunSpec,
    engine: EngineSource,
    sandbox: Sandbox,
    prepared: PreparedArm,
    environment: dict[str, str],
    *,
    stage: str,
    split: JudgedSet,
    log: _Log,
) -> Measurement:
    """Run the app's harness once, for one stage and one split."""

    report_path = prepared.report_path.with_name(
        REPORT_PATTERN.format(split=segment(split.name))
    )
    command = _measure_command(spec, engine, sandbox, split, report_path, stage=stage)
    log.emit(f"{stage}[{split.name}]: {command}")
    outcome = _run(
        command, environment, cwd=sandbox.root, timeout=MEASURE_TIMEOUT_SECONDS
    )
    return Measurement(
        stage=stage,
        split=split.name,
        command=command,
        exit_code=int(outcome["exit_code"]),
        elapsed_seconds=float(outcome["elapsed_seconds"]),
        stdout=str(outcome["stdout"]),
        stderr=str(outcome["stderr"]),
        report=report_path if stage == "measure" else None,
    )


def _with_facts(item: Measurement, facts: ReportFacts) -> Measurement:
    return Measurement(
        stage=item.stage,
        split=item.split,
        command=item.command,
        exit_code=item.exit_code,
        elapsed_seconds=item.elapsed_seconds,
        stdout=item.stdout,
        stderr=item.stderr,
        report=item.report,
        report_facts=facts.describe(),
    )


def _child_environment(engine: EngineSource, sandbox: Sandbox) -> dict[str, str]:
    from ..engine.runner import child_environment

    return child_environment(engine, config_home=sandbox.config_home)


def locate_or_refuse(root: Path) -> EngineSource:
    from ..engine.locate import locate_engine

    try:
        return locate_engine(root)
    except ExperimentError as exc:
        raise ExperimentError(
            f"The checkout at {root} is not a research-rag tree this harness can "
            f"measure: {exc}"
        ) from exc


def effective_overlay(arm: Arm, harness: dict[str, Any]) -> dict[str, Any]:
    """An arm's settings, including the ones its `harness` block implies.

    A specification can set a setting twice: once in an arm's `overlay` and once
    through a `harness` entry that is really a setting. The overlay wins, because
    the overlay is the arm-specific statement and the `harness` block is what
    every arm of the run shares.
    """

    overlay: dict[str, Any] = {}
    for key, setting in HARNESS_SETTINGS.items():
        value = harness.get(key)
        if value is not None:
            overlay[setting] = value
    overlay.update(arm.overlay)
    return overlay


def _measure_command(
    spec: RunSpec,
    engine: EngineSource,
    sandbox: Sandbox,
    split: JudgedSet,
    report_path: Path,
    *,
    stage: str,
) -> list[str]:
    """The exact command the app's own harness is asked to run.

    The harness is named from the engine tree rather than from the installed
    distribution, because an arm measures the engine under test and a script from
    somewhere else would be a second implementation of the same measurement. The
    corpus, the judged set, and the report all live outside the engine tree.
    """

    command = [
        sys.executable,
        str(engine.harness),
        PROJECT_FLAG,
        str(sandbox.root),
        JUDGMENTS_FLAG,
        str(split.path),
        REPORT_FLAG,
        str(report_path),
    ]
    for key, value in spec.harness.items():
        command.extend(_harness_flag(key, value))
    if stage == "validate":
        command.append(VALIDATE_FLAG)
    return command


def _harness_flag(key: str, value: Any) -> list[str]:
    """Turn one `harness` entry into the flags the app's harness accepts.

    The flag is the key with underscores turned into hyphens, which is the
    spelling the app's own parser uses. A list is a comma-separated value for the
    entries declared in `HARNESS_JOINED` and a repeated flag for the ones declared
    in `HARNESS_REPEATED`; a list anywhere else is refused, because the only two
    shapes the app's harness has are those and guessing which one was meant
    measures something other than what the specification says.
    """

    flag = f"--{key.replace('_', '-')}"
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
        if not items:
            return []
        if key in HARNESS_JOINED:
            return [flag, ",".join(items)]
        if key in HARNESS_REPEATED:
            arguments: list[str] = []
            for item in items:
                arguments.extend([flag, item])
            return arguments
        raise ExperimentError(
            f"The harness entry {key!r} takes one value, not a list. Allowed "
            f"list-valued entries: {sorted(HARNESS_JOINED | HARNESS_REPEATED)}."
        )
    if value is None:
        return []
    return [flag, str(value)]


def _run(
    command: list[str],
    environment: dict[str, str],
    *,
    cwd: Path,
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            cwd=str(cwd),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": -1,
            "elapsed_seconds": time.perf_counter() - started,
            "stdout": _text(exc.stdout),
            "stderr": f"timed out after {timeout} s: {exc}",
            "command": command,
        }
    except OSError as exc:
        raise ExperimentError(f"Cannot run {command[0]}: {exc}") from exc
    return {
        "exit_code": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "command": command,
    }


def _interpolate(
    command: str | list[str], sandbox: Sandbox, engine: EngineSource
) -> list[str]:
    """Expand the tokens a prepare command may use, and only those.

    A command written as one string is split the way a shell would, so quoting in a
    specification means what it says. A command written as a list is already split,
    so its elements are expanded one by one and nothing is re-quoted: a path with a
    space in it stays one argument.

    An unknown token is left as it is written. A command that expected a
    substitution the harness cannot make then fails inside the command, naming the
    token, rather than the harness quietly measuring a path that happens to be there.

    `{source}` is the copy's own originals. The original project's path is never
    substituted into a command, because a command that could write there is a
    command that would fail the guard for the run that tried it.
    """

    import shlex

    values = {
        "{project}": str(sandbox.root),
        "{source}": str(sandbox.source_root),
        "{tree}": str(engine.root),
        "{sandbox}": str(sandbox.workspace),
    }

    def expand(text: str) -> str:
        for token, value in values.items():
            text = text.replace(token, value)
        return text

    if isinstance(command, str):
        return shlex.split(expand(command))
    return [expand(str(part)) for part in command]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _digest_of(path: Path) -> str:
    import hashlib

    accumulator = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            accumulator.update(block)
    return accumulator.hexdigest()


def _tail(text: str, count: int = 20) -> str:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(lines[-count:]) if lines else ""


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class _Log:
    """An arm's log, appended and flushed as the arm runs.

    The log is the only thing that survives an interruption, so it is written as
    each line and each subprocess block is finished rather than accumulated and
    written at the end of the arm.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def outcome(self, outcome: dict[str, Any]) -> None:
        self.emit(f"exit {outcome['exit_code']} in {outcome['elapsed_seconds']:.1f} s")
        self.output_text("stdout", str(outcome.get("stdout") or ""))
        self.output_text("stderr", str(outcome.get("stderr") or ""))

    def output(self, item: Measurement) -> None:
        self.emit(f"exit {item.exit_code} in {item.elapsed_seconds:.1f} s")
        self.output_text("stdout", item.stdout)
        self.output_text("stderr", item.stderr)

    def output_text(self, label: str, text: str) -> None:
        self.emit(f"--- {label} ---")
        if text.strip():
            self.emit(text.rstrip("\n"))
        self.emit(f"--- end {label} ---")


__all__ = [
    "ARM_EVIDENCE",
    "HARNESS_JOINED",
    "HARNESS_REPEATED",
    "HARNESS_SETTINGS",
    "PREPARE_TIMEOUT_SECONDS",
    "ArmFailure",
    "ArmResult",
    "PreparedArm",
    "effective_overlay",
    "prepare_arm",
    "requested_modes",
    "run_arm",
]
