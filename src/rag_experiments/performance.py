"""Run balanced app-owned performance probes on guarded disposable projects."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from . import resource_limits
from .engine.locate import locate_engine, tree_digest
from .engine.runner import child_environment
from .errors import ExperimentError
from .experiment.spec import load_spec
from .niceness import run_low_priority
from .report.record import close_guard, take_guard
from .run import prepare_experiment


def _write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preflight(spec_path: Path, app_source: Path, blocks: int, warmup_count: int):
    if type(blocks) is not int or not 2 <= blocks <= 20:
        raise ExperimentError("Use 2..20 repeated performance blocks")
    if type(warmup_count) is not int or not 1 <= warmup_count <= 30:
        raise ExperimentError("Use 1..30 explicit warm-up searches")
    spec = load_spec(spec_path)
    if spec.harness.get("offline") is not True:
        raise ExperimentError(
            "Performance probes require explicit offline cached-model operation"
        )
    if any(arm.kind != "settings" or arm.prepare for arm in spec.arms):
        raise ExperimentError(
            "Performance probes use settings-only query trials without preparation commands"
        )
    if (
        spec.harness.get("modes") != ["hybrid+rerank"]
        or spec.harness.get("deep_top_k") != 0
    ):
        raise ExperimentError(
            "Performance protocol measures only hybrid+rerank, with no deep pass"
        )
    engine = locate_engine(app_source)
    if engine.dirty or engine.untracked:
        raise ExperimentError(
            "Benchmark a clean, pinned app checkout, not concurrently edited source"
        )
    script = engine.root / "scripts" / "benchmark_retrieval.py"
    if not script.is_file():
        raise ExperimentError("The pinned app has no benchmark_retrieval.py probe")
    return spec, engine, script


def run_performance(
    spec_path: Path,
    app_source: Path,
    workspace: Path,
    runs: Path,
    *,
    blocks=3,
    warmup_count=3,
    seed=0,
) -> dict:
    """Retain logs and guards; the app owns every latency/resource calculation."""

    spec, engine, script = _preflight(spec_path, app_source, blocks, warmup_count)
    prepared = prepare_experiment(
        spec, app_source=engine, workspace=workspace, runs_directory=runs
    )
    if not prepared.prepared:
        raise ExperimentError(
            f"Performance preparation failed; see {prepared.record_path}: {prepared.error}"
        )
    directory = prepared.run_directory / "performance"
    directory.mkdir(mode=0o700)
    queries = {}
    for kept in prepared.record["judgments"]:
        path = Path(kept["kept_at"])
        if _sha(path) != kept["kept_sha256"]:
            raise ExperimentError("Prepared query input changed")
        for query in json.loads(path.read_text())["queries"]:
            qid, text = query["query_id"], query["query"]
            if qid in queries and queries[qid] != text:
                raise ExperimentError("Partitions disagree about a benchmark question")
            queries[qid] = text
    if not queries:
        raise ExperimentError("No benchmark queries were provided")
    guard_before = None
    engine_before = None
    samples = []
    error = None
    interruption = None
    summary_path = directory / "summary.json"
    orders = []
    try:
        guard_before = take_guard(spec.source_project)
        engine_before = tree_digest(engine.root)
        if engine_before is None:
            raise ExperimentError("The pinned engine could not be fingerprinted")
        arms = prepared.results
        start = list(range(len(arms)))
        random.Random(seed).shuffle(start)
        for block in range(blocks):
            order = start[block % len(start) :] + start[: block % len(start)]
            orders.append([arms[position].arm.name for position in order])
            for position in order:
                result = arms[position]
                input_path = directory / f"block-{block}-{result.arm.name}-input.json"
                report = directory / f"block-{block}-{result.arm.name}.json"
                log = directory / f"block-{block}-{result.arm.name}.log"
                generation = result.sandbox.generation_ids[0]
                _write(
                    input_path,
                    {
                        "schema_version": 1,
                        "block": block,
                        "condition_id": result.arm.name,
                        "generation_id": generation,
                        "queries": [
                            {"query_id": qid, "query": queries[qid]}
                            for qid in sorted(queries)
                        ],
                        "metadata": {
                            "role": "performance_only",
                            "engine": engine.describe(),
                            "settings": result.settings,
                            "query_order_seed": seed + block,
                        },
                    },
                )
                command = [
                    sys.executable,
                    "-B",
                    str(script),
                    "--project",
                    str(result.sandbox.root),
                    "--queries",
                    str(input_path),
                    "--report",
                    str(report),
                    "--warmup-count",
                    str(warmup_count),
                    "--seed",
                    str(seed + block),
                ]
                environment = child_environment(
                    engine, config_home=result.sandbox.config_home
                )
                outcome = run_low_priority(
                    command,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,
                )
                log.write_text(outcome.stdout + "\n" + outcome.stderr, encoding="utf-8")
                log.chmod(0o600)
                sample = {
                    "block": block,
                    "condition_id": result.arm.name,
                    "command": command,
                    "input": str(input_path),
                    "input_sha256": _sha(input_path),
                    "report": str(report),
                    "log": str(log),
                    "exit_code": outcome.returncode,
                }
                if report.is_file():
                    sample["sha256"] = _sha(report)
                samples.append(sample)
                if outcome.returncode or not report.is_file():
                    raise ExperimentError(f"Performance probe failed; see {log}")
        list_path = directory / "reports.json"
        _write(
            list_path,
            {
                "reports": [{**sample, "path": sample["report"]} for sample in samples],
                "condition_orders": orders,
                "baseline_condition_id": spec.arms[0].name,
            },
        )
        command = [
            sys.executable,
            "-B",
            str(script),
            "--summarize",
            str(list_path),
            "--report",
            str(summary_path),
        ]
        outcome = run_low_priority(
            command,
            env=child_environment(engine),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        (directory / "summary.log").write_text(
            outcome.stdout + "\n" + outcome.stderr, encoding="utf-8"
        )
        if outcome.returncode or not summary_path.is_file():
            raise ExperimentError(
                "App performance summarizer refused the reports; see summary.log"
            )
    except BaseException as exc:  # noqa: BLE001 - finalize every refusal or interrupt before re-raising
        error = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interruption = exc
    finally:
        try:
            guard = close_guard(guard_before) if guard_before is not None else None
            guard_info = (
                guard.describe()
                if guard is not None
                else {"state": "unknown", "error": "initial guard unavailable"}
            )
        except Exception as exc:  # noqa: BLE001 - failed guard means unknown, never healthy
            guard = None
            guard_info = {"state": "unknown", "error": f"{type(exc).__name__}: {exc}"}
        try:
            engine_after = tree_digest(engine.root)
        except Exception:  # noqa: BLE001 - unavailable code fingerprint invalidates the run
            engine_after = None
        verdict = "failed" if error else "verified"
        if guard is None or engine_before is None or engine_after is None:
            verdict = "incomplete"
        elif not guard.unchanged or engine_before != engine_after:
            verdict = "inputs_changed"
        record = {
            "schema_version": 1,
            "protocol": "app_performance_probe_v1",
            "verdict": verdict,
            "preparation": str(prepared.record_path),
            "source_guard": guard_info,
            "engine_sha256_before": engine_before,
            "engine_sha256_after": engine_after,
            "blocks": blocks,
            "warmup_count": warmup_count,
            "condition_orders": orders,
            "samples": samples,
            "resource_limits": resource_limits.describe(),
            "error": error,
            "performance_only": True,
            "summary": str(summary_path) if summary_path.is_file() else None,
        }
        if summary_path.is_file():
            record["summary_sha256"] = _sha(summary_path)
        _write(directory / "performance-run.json", record)
    if interruption is not None:
        raise interruption
    if verdict != "verified":
        raise ExperimentError(
            f"Performance run is {verdict}: {error}; {directory / 'performance-run.json'}"
        )
    return {
        "verdict": verdict,
        "record": str(directory / "performance-run.json"),
        "summary": str(summary_path),
        "blocks": blocks,
        "conditions": len(prepared.results),
        "unique_queries": len(queries),
        "performance_only": True,
        "policy_ready": False,
    }
