"""Run one arm, and return what it produced.

An arm is measured by running the app's own evaluation harness as a subprocess,
against a sandbox, through an engine tree. Nothing is computed here: the harness
writes the report, the harness writes the numbers, and this module only says what
it ran, what it returned, and how long it took.

An arm fails loudly. A non-zero exit, a missing report, and an unparsed report are
all refusals rather than a row of zeroes, because a row of zeroes read as "this
variant found nothing" when it in fact measured nothing would be the most
expensive kind of wrong number this toolkit could produce.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine.locate import EngineSource, locate_engine
from ..engine.runner import resolve_settings_for_tree
from ..errors import ExperimentError
from ..sandbox import Sandbox, pin_settings
from ..sandbox import create as create_sandbox
from .checkout import Checkout, make_checkout
from .spec import (
    HARNESS_JOINED,
    HARNESS_REPEATED,
    HARNESS_SETTINGS,
    Arm,
    JudgedSet,
    RunSpec,
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
#: segment, so this is the only place a name is turned into a file name.
REPORT_PATTERN = "report-{split}.json"


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
        return record


@dataclass(frozen=True, slots=True)
class ArmResult:
    """What one arm did, and where its evidence is."""

    arm: Arm
    engine: EngineSource
    checkout: Checkout | None
    sandbox: Sandbox
    settings: dict[str, Any]
    prepare: list[dict[str, Any]]
    validate: tuple[Measurement, ...]
    measure: tuple[Measurement, ...]
    log_path: Path

    def reports(self) -> dict[str, Path]:
        """Where each split's report was written, keyed by split name."""

        return {
            item.split: item.report for item in self.measure if item.report is not None
        }

    def describe(self) -> dict[str, Any]:
        return {
            "arm": self.arm.describe(),
            "engine": self.engine.describe(),
            "checkout": None if self.checkout is None else self.checkout.describe(),
            "sandbox": self.sandbox.describe(),
            "settings": self.settings,
            "prepare": self.prepare,
            "validate": [item.describe() for item in self.validate],
            "measure": [item.describe() for item in self.measure],
            "reports": {name: str(path) for name, path in self.reports().items()},
            "log": str(self.log_path),
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
    log_path: Path
    report_path: Path

    def environment(self) -> dict[str, str]:
        from ..engine.runner import child_environment

        return child_environment(self.engine, config_home=self.sandbox.config_home)

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.engine.describe(),
            "checkout": None if self.checkout is None else self.checkout.describe(),
            "sandbox": self.sandbox.describe(),
            "settings": self.settings,
            "log": str(self.log_path),
        }


def prepare_arm(
    spec: RunSpec,
    arm: Arm,
    *,
    workspace: Path,
    run_directory: Path,
    app_source: EngineSource,
) -> PreparedArm:
    """Materialize an arm's sandbox, its engine, and its pinned settings.

    Each arm gets its own sandbox. That is the isolation the plan asks for: a
    preparation step that rebuilds a generation writes into the arm's copy and
    cannot be seen by the arm measured after it.
    """

    area = workspace / arm.name
    area.mkdir(parents=True, exist_ok=True)
    log_path = area / "arm.log"
    output: list[str] = []

    def emit(line: str) -> None:
        output.append(line)
        log_path.write_text("\n".join(output) + "\n", encoding="utf-8")

    sandbox = create_sandbox(
        workspace,
        name=f"{spec.name}-{arm.name}",
        source=spec.source_project,
        generations=spec.generations,
    )

    checkout: Checkout | None = None
    engine = app_source
    if arm.kind == "code":
        tree_root = area / "engine"
        emit(f"checkout: {app_source.root} -> {tree_root}")
        checkout = make_checkout(
            tree_root, app_source=app_source, base=arm.base, patch=arm.patch
        )
        engine = locate_engine(tree_root)

    overlay = effective_overlay(arm, spec.harness)
    emit(f"resolving settings in {engine.root} with overlay {overlay or '{}'}")
    answer = resolve_settings_for_tree(
        engine,
        project=spec.source_project,
        overlay=overlay,
        config_home=sandbox.config_home,
    )
    pin_settings(sandbox, answer.document)
    settings = answer.describe()
    settings["written_to"] = str(sandbox.settings_file)
    settings["config_home"] = str(sandbox.config_home)
    emit(
        f"pinned {len(answer.values)} settings to {sandbox.settings_file} "
        f"({len(overlay)} overridden)"
    )
    return PreparedArm(
        arm=arm,
        engine=engine,
        checkout=checkout,
        sandbox=sandbox,
        settings=settings,
        log_path=log_path,
        report_path=area / "report.json",
    )


def run_arm(
    spec: RunSpec,
    arm: Arm,
    *,
    workspace: Path,
    run_directory: Path,
    app_source: EngineSource,
    keep_sandbox: bool = True,
    validate_first: bool = True,
) -> ArmResult:
    """Prepare one arm, optionally prepare the corpus, measure it, and report.

    `validate_first` runs the app's harness in its validation stage before any
    search, so a judged target that cannot be resolved stops the run before it
    costs an hour of inference rather than after.
    """

    prepared = prepare_arm(
        spec,
        arm,
        workspace=workspace,
        run_directory=run_directory,
        app_source=app_source,
    )
    sandbox = prepared.sandbox
    engine = prepared.engine
    log_path = prepared.log_path
    output: list[str] = log_path.read_text(encoding="utf-8").splitlines()

    def emit(line: str) -> None:
        output.append(line)
        log_path.write_text("\n".join(output) + "\n", encoding="utf-8")

    environment = prepared.environment()

    prepare_results: list[dict[str, Any]] = []
    for command in arm.prepare:
        arguments = _interpolate(command, sandbox, engine)
        emit(f"prepare: {arguments}")
        outcome = _run(
            arguments, environment, cwd=sandbox.root, timeout=PREPARE_TIMEOUT_SECONDS
        )
        outcome["command"] = arguments
        prepare_results.append(outcome)
        emit(
            f"prepare exit {outcome['exit_code']} in {outcome['elapsed_seconds']:.1f} s"
        )
        if outcome["exit_code"] != 0:
            _fail(arm, log_path, "prepare", outcome)

    # The app's own harness resolves a judged target before it searches, and a
    # target it cannot resolve uniquely makes every number for that split
    # meaningless. That check runs first because it costs no inference and finds
    # the failure in seconds rather than after an hour of cross-encoding.
    validate: list[Measurement] = []
    if validate_first:
        for split in spec.judgments:
            validate.append(
                _invoke(
                    spec,
                    engine,
                    sandbox,
                    prepared,
                    environment,
                    stage="validate",
                    split=split,
                    emit=emit,
                )
            )
            item = validate[-1]
            emit(
                f"validate[{split.name}] exit {item.exit_code} in "
                f"{item.elapsed_seconds:.1f} s"
            )
            if item.exit_code != 0:
                _fail(
                    arm,
                    log_path,
                    f"validate ({split.name})",
                    _outcome(item),
                    (
                        f"The judged set {split.path} could not be resolved against "
                        f"this arm's generation. Every number for this split would "
                        f"describe nothing, so the run stops here rather than after "
                        f"measuring it."
                    ),
                )

    measurements: list[Measurement] = []
    for split in spec.judgments:
        measurements.append(
            _invoke(
                spec,
                engine,
                sandbox,
                prepared,
                environment,
                stage="measure",
                split=split,
                emit=emit,
            )
        )
        item = measurements[-1]
        emit(
            f"measure[{split.name}] exit {item.exit_code} in {item.elapsed_seconds:.1f} s"
        )
        if item.exit_code != 0:
            _fail(arm, log_path, f"measure ({split.name})", _outcome(item))
        _require_report(arm, item, log_path)

    if not keep_sandbox:
        from ..sandbox import remove as remove_sandbox

        remove_sandbox(workspace, sandbox.name)

    return ArmResult(
        arm=arm,
        engine=prepared.engine,
        checkout=prepared.checkout,
        sandbox=sandbox,
        settings=prepared.settings,
        prepare=prepare_results,
        validate=tuple(validate),
        measure=tuple(measurements),
        log_path=log_path,
    )


def _invoke(
    spec: RunSpec,
    engine: EngineSource,
    sandbox: Sandbox,
    prepared: PreparedArm,
    environment: dict[str, str],
    *,
    stage: str,
    split: JudgedSet,
    emit: Callable[[str], None],
) -> Measurement:
    """Run the app's harness once, for one stage and one split."""

    report_path = prepared.report_path.with_name(
        REPORT_PATTERN.format(split=_segment(split.name))
    )
    command = _measure_command(spec, engine, sandbox, split, report_path, stage=stage)
    emit(f"{stage}[{split.name}]: {command}")
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


def _require_report(arm: Arm, item: Measurement, log_path: Path) -> None:
    """Refuse a clean exit that wrote no readable report.

    A harness that resolved nothing and exited cleanly has measured nothing, and a
    row of zeroes read as "this variant found nothing" is the most expensive kind
    of wrong number a harness can produce.
    """

    if item.report is None or not item.report.is_file():
        raise ExperimentError(
            f"Arm {arm.name!r} exited cleanly and wrote no report for split "
            f"{item.split!r} at {item.report}. The full output is in {log_path}."
        )
    try:
        json.loads(item.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(
            f"The report arm {arm.name!r} wrote for split {item.split!r} is not "
            f"readable JSON: {item.report}: {exc}. The full output is in {log_path}."
        ) from exc


def _segment(name: str) -> str:
    """Turn a split name into one path segment.

    A split name is a record key and a printed label, so it is not assumed to be a
    safe file name; anything outside a plain word is replaced rather than
    interpreted as a path.
    """

    safe = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in name
    ).strip("-")
    return safe or "split"


def _outcome(item: Measurement) -> dict[str, Any]:
    return {
        "exit_code": item.exit_code,
        "elapsed_seconds": item.elapsed_seconds,
        "stdout": item.stdout,
        "stderr": item.stderr,
        "command": list(item.command),
    }


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


def _fail(
    arm: Arm,
    log_path: Path,
    stage: str,
    outcome: dict[str, Any],
    because: str = "",
) -> None:
    tail = _tail(outcome.get("stderr") or outcome.get("stdout") or "")
    raise ExperimentError(
        f"Arm {arm.name!r} failed at the {stage} step with exit "
        f"{outcome['exit_code']}.\n"
        + (f"  because:  {because}\n" if because else "")
        + f"  command: {' '.join(outcome['command'])}\n"
        f"  output:  {tail}\n"
        f"  full:    {log_path}"
    )


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


def _interpolate(command: str, sandbox: Sandbox, engine: EngineSource) -> list[str]:
    """Expand the tokens a prepare command may use, and only those.

    An unknown token is left as it is written. A command that expected a
    substitution the harness cannot make then fails inside the command, naming
    the token, rather than the harness quietly measuring a path that happens to
    be there.
    """

    import shlex

    values = {
        "{project}": str(sandbox.root),
        "{source}": str(sandbox.source_root),
        "{tree}": str(engine.root),
        "{sandbox}": str(sandbox.workspace),
    }
    expanded = command
    for token, value in values.items():
        expanded = expanded.replace(token, value)
    return shlex.split(expanded)


def _tail(text: str, count: int = 20) -> str:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(lines[-count:]) if lines else ""


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


__all__ = [
    "HARNESS_JOINED",
    "HARNESS_REPEATED",
    "HARNESS_SETTINGS",
    "PREPARE_TIMEOUT_SECONDS",
    "ArmResult",
    "PreparedArm",
    "effective_overlay",
    "prepare_arm",
    "run_arm",
]
