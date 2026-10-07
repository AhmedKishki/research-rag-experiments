"""Delegate saved-label scoring to the app; calculate no quality metric here."""

import subprocess
import sys
from pathlib import Path

from ..errors import ExperimentError
from ..niceness import run_low_priority
from ..sandbox.layout import refuse_nested
from .build import _private_output
from .export import HANDOFF_PROTOCOL
from .inputs import digest, read_json, safe_file


def score_handoff(
    source: Path,
    app_source: Path,
    report: Path,
    *,
    baseline=None,
    bootstrap_samples=2000,
    seed=0,
):
    source = safe_file(source)
    app_source = app_source.expanduser().absolute()
    script = safe_file(app_source / "scripts" / "evaluate_pooled.py")
    report = report.expanduser().absolute()
    handoff = read_json(source)
    if handoff.get("protocol") != HANDOFF_PROTOCOL:
        raise ExperimentError("The app-owned scorer requires an author_pool_v1 handoff")
    if (
        report.exists()
        or report.is_symlink()
        or report == source
        or any(p.is_symlink() for p in report.parents)
    ):
        raise ExperimentError("Pooled report must use a new nonlinked path")
    for root in [
        app_source,
        *map(Path, handoff.get("provenance", {}).get("protected_roots", [])),
    ]:
        refuse_nested(
            report, root, what="pooled report", inside="original or retained source"
        )
    _private_output(report.parent / ".score-output")
    input_sha, script_sha = digest(source), digest(script)
    command = [
        sys.executable,
        "-I",
        "-B",
        str(script),
        "--input",
        str(source),
        "--report",
        str(report),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--seed",
        str(seed),
    ]
    if baseline is not None:
        command.extend(["--baseline", baseline])
    try:
        completed = run_low_priority(
            command, capture_output=True, text=True, timeout=300, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExperimentError(
            f"App-owned pooled scoring could not finish: {exc}"
        ) from exc
    if completed.returncode:
        raise ExperimentError(
            f"App-owned scorer refused the handoff: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    if digest(source) != input_sha or digest(script) != script_sha:
        raise ExperimentError(
            f"Scoring inputs or app scorer changed during execution; report is not verified: {report}"
        )
    data = read_json(report)
    if data.get("report_meta", {}).get("protocol") != HANDOFF_PROTOCOL:
        raise ExperimentError("App scorer wrote an unexpected protocol")
    return {
        "protocol": HANDOFF_PROTOCOL,
        "report": str(report),
        "sha256": digest(report),
        "handoff_sha256": input_sha,
        "scorer_sha256": script_sha,
        "conditions": len(data.get("conditions", [])),
        "role": "exploratory",
        "policy_ready": False,
    }
