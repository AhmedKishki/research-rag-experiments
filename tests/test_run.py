"""That a run does what it says, and leaves a record when it cannot finish.

These are the tests for the run itself rather than for one module: every case drives
`run_experiment` with a real engine tree, a real sandbox copy, and a real
subprocess per harness invocation. The harness is a stub script, because measuring a
corpus needs the whole retrieval stack; what is under test is everything around it —
what the run hands the harness, what it writes where, and what a record says when an
arm refuses, the source project moves, or the terminal is closed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from conftest import (
    FIRST_GENERATION,
    PROJECT_ID,
    build_judged_set,
    build_project,
)

from rag_experiments.engine.locate import locate_engine
from rag_experiments.errors import ExperimentError
from rag_experiments.experiment.spec import load_spec
from rag_experiments.report.record import (
    VERDICT_ENGINE_CHANGED,
    VERDICT_FAILED,
    VERDICT_INCOMPLETE,
    VERDICT_INTERRUPTED,
    VERDICT_NO_ARMS,
    VERDICT_SOURCE_CHANGED,
    VERDICT_VERIFIED,
)
from rag_experiments.run import run_experiment

#: The stub harness stands in for the app's own evaluation script. It reads the
#: flags the run builds, checks that the corpus it was handed is a copy rather than
#: the original, and writes a report in the shape the app's harness writes, because
#: everything downstream of it is checked against that shape.
STUB_HARNESS = '''\
"""A stand-in for the app's evaluation harness, written to fail on purpose.

It reads the judged file it is handed, applies the selection the same way the app's
harness does, and writes one row per selected query per mode with that query's own
identity. That is the contract the report is checked against, so a fixture that
wrote a made-up count or an empty query list would be testing a report nobody would
ever produce.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RERANK_MODE = "hybrid+rerank"
QUERY_CLASSES = ("quote", "paraphrase", "entity")
MODES = ("bm25", "dense", "hybrid", RERANK_MODE)
REPORT_SCHEMA_VERSION = 2


def _names(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--deep-top-k", type=int, default=0)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--deep-modes", default="")
    parser.add_argument("--classes", default=",".join(QUERY_CLASSES))
    parser.add_argument("--skip-targets", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reranker-model", action="append")
    parser.add_argument("--allow-degraded-rerank", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    print(f"harness: project {project}")
    print(f"harness: judged {args.judgments}")
    print(f"harness: top_k {args.top_k} modes {args.modes}")

    # A measurement of the original project would defeat every guarantee this
    # harness makes, so the stub refuses one outright.
    if project == Path(os.environ["RAG_EXPERIMENTS_STUB_ORIGINAL"]).resolve():
        print("harness: refusing to measure the source project", file=sys.stderr)
        return 3

    settings = project / ".research-rag" / "config.toml"
    print(f"harness: pinned settings at {settings}")
    if not settings.is_file():
        print("harness: no pinned settings file", file=sys.stderr)
        return 4

    behaviour = os.environ.get("RAG_EXPERIMENTS_STUB_BEHAVIOUR", "")
    if behaviour == "validate-refuses" and args.validate_only:
        print("harness: target t01 resolved to 0 chunks", file=sys.stderr)
        return 2
    if behaviour == "measure-refuses" and not args.validate_only:
        print("harness: reranker model not installed", file=sys.stderr)
        return 2
    if behaviour == "interrupted" and not args.validate_only:
        os.kill(os.getppid(), 2)  # SIGINT: the run's own process, mid-measurement
        print("harness: signalled the run", file=sys.stderr)
        return 130
    if behaviour == "no-report" and not args.validate_only:
        print("harness: exiting cleanly without writing a report")
        return 0

    judged = json.loads(args.judgments.read_text(encoding="utf-8"))
    classes = _names(args.classes) or list(QUERY_CLASSES)
    skipped = set(_names(args.skip_targets))
    selected = [
        query
        for query in judged["queries"]
        if query["class"] in classes and query["target_id"] not in skipped
    ]
    if args.limit:
        selected = selected[: args.limit]
    print(
        f"harness: {len(selected)} of {len(judged['queries'])} queries selected"
        f" (classes {','.join(classes)}, skip {','.join(sorted(skipped)) or 'none'}"
        f", limit {args.limit or 'none'})"
    )
    if not selected:
        print("harness: No queries selected", file=sys.stderr)
        return 2

    if args.validate_only:
        print("harness: every judged target resolved uniquely; no search was run.")
        return 0

    modes = _names(args.modes)
    deep_modes = _names(args.deep_modes) if args.deep_top_k else []
    by_target = {target["target_id"]: target for target in judged["targets"]}
    runs = []
    for query in selected:
        target = by_target[query["target_id"]]
        for mode in modes:
            runs.append(
                {
                    "query_id": query["query_id"],
                    "target_id": query["target_id"],
                    "class": query["class"],
                    "query": query["query"],
                    "mode": mode,
                    "top_k": args.top_k,
                    "rank": 1,
                    "success_at_1": True,
                    "success_at_3": True,
                    "success_at_k": True,
                    "reciprocal_rank": 1.0,
                    "ndcg_at_k": 1.0,
                    "document_success_at_k": True,
                    "result_count": 1,
                    "distinct_source_count": 1,
                    "returned_chunk_ids": [target["chunk_id_at_measurement"] or "c01"],
                    "candidate_count": 1,
                    "candidate_depth": min(200, args.top_k * 20),
                    "rerank_window": 60 if mode.startswith(RERANK_MODE) else 0,
                    "rerank_requested": mode.startswith(RERANK_MODE),
                    "reranked": mode.startswith(RERANK_MODE),
                    "rerank_fallback": None,
                    "elapsed_seconds": 0.5,
                }
            )
    deep_runs = [
        {**run, "mode": mode, "top_k": args.deep_top_k}
        for run in runs
        for mode in deep_modes
        if run["mode"] == mode
    ]

    def summarize(rows: list[dict[str, object]], top_k: int) -> dict[str, object]:
        summary: dict[str, object] = {}
        for label in dict.fromkeys(str(row["mode"]) for row in rows):
            chosen = [row for row in rows if row["mode"] == label]
            summary[label] = {
                "overall": {
                    "query_count": len(chosen),
                    "top_k": top_k,
                    "mrr": 1.0,
                    "ndcg_at_k": 1.0,
                    "document_success_at_k": 1.0,
                    "p50_seconds": 0.5,
                }
            }
        return summary

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "judgments": {
            "path": str(args.judgments),
            "schema_version": judged.get("schema_version"),
            "protocol": judged.get("protocol"),
            "query_count": len(judged["queries"]),
            "evaluated_query_count": len(selected),
            "target_count": len(judged["targets"]),
        },
        "settings": {
            "selected_modes": modes,
            "selected_deep_modes": deep_modes,
            "selected_classes": classes,
            "reranker_models": list(args.reranker_model or []),
            "skipped_targets": sorted(skipped),
            "top_k": args.top_k,
            "deep_top_k": args.deep_top_k,
            "include_staleness": False,
            "allow_degraded_rerank": bool(args.allow_degraded_rerank),
        },
        "runs": runs,
        "deep_runs": deep_runs,
        "summary": summarize(runs, args.top_k),
        "deep_summary": summarize(deep_runs, args.deep_top_k),
        "timing": {
            "total_seconds": 0.5,
            "search_count": len(runs) + len(deep_runs),
        },
    }
    if behaviour == "malformed-report":
        args.report.write_text("{half a file", encoding="utf-8")
        return 0
    if behaviour == "wrong-depth":
        report["settings"]["top_k"] = 50
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
    print(f"harness: {len(runs)} searches; report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


@pytest.fixture
def stub_engine(tmp_path: Path, app_source: Path) -> Path:
    """A copy of the engine tree whose harness script is a stub.

    Everything else is the real app, so the settings resolution, the registry, and
    the layer stack a run uses are the ones a real run uses. Only the harness is
    replaced, because that is the part that would need a corpus and an hour of
    inference. It is a copy rather than a link, so a write into the tree under test
    is visible here and the tests can hold that nothing wrote to it.
    """

    import shutil

    tree = tmp_path / "stub-engine"
    shutil.copytree(
        app_source,
        tree,
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", "*.pyc"),
        symlinks=True,
    )
    (tree / "scripts" / "evaluate_retrieval.py").write_text(
        STUB_HARNESS, encoding="utf-8"
    )
    return tree


#: A package that finds the installed app's modules while being its own package.
#: A code arm clones its engine, so the tree under test must be a real repository
#: with real directories; this is how the settings resolution still reaches the real
#: app's registry and layer stack from inside such a tree.
PATH_EXTENDING_INIT = """\
'''A fixture engine package that reads the installed app's modules.'''

import os

__path__.append(
    os.path.join(os.environ["RAG_EXPERIMENTS_STUB_APP_SRC"], "research_rag")
)
"""


@pytest.fixture
def git_engine(
    tmp_path: Path, app_source: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """An engine tree that is a real git repository, whose harness is the stub.

    A code arm clones the tree it measures, so the tree has to be a repository with
    real directories rather than links into a working checkout. The package extends
    its own path to the installed app, so the settings a run pins are resolved by
    the real code.
    """

    monkeypatch.setenv("RAG_EXPERIMENTS_STUB_APP_SRC", str(app_source / "src"))
    tree = tmp_path / "git-engine"
    (tree / "scripts").mkdir(parents=True)
    (tree / "src" / "research_rag").mkdir(parents=True)
    (tree / "src" / "research_rag" / "__init__.py").write_text(
        PATH_EXTENDING_INIT, encoding="utf-8"
    )
    (tree / "scripts" / "evaluate_retrieval.py").write_text(
        STUB_HARNESS, encoding="utf-8"
    )
    (tree / "README.md").write_text("fixture engine\n", encoding="utf-8")
    for arguments in (
        ["init", "-q"],
        ["config", "user.email", "tests@example.com"],
        ["config", "user.name", "tests"],
        ["add", "-A"],
        ["commit", "-q", "-m", "fixture engine"],
    ):
        subprocess.run(
            ["git", "-C", str(tree), *arguments], check=True, capture_output=True
        )
    return tree


@pytest.fixture(autouse=True)
def _no_engine_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RESEARCH_RAG_APP_SOURCE", "RESEARCH_RAG_MODEL_CACHE_ROOT"):
        monkeypatch.delenv(name, raising=False)


def _spec_file(
    tmp_path: Path,
    project: Path,
    *,
    arms: list[dict[str, object]],
    harness: dict[str, object] | None = None,
    judged: str = "judged.json",
    splits: list[dict[str, str]] | None = None,
    name: str = "fixture",
) -> Path:
    judged_path = tmp_path / judged
    if not judged_path.is_file():
        build_judged_set(judged_path)
    path = tmp_path / "spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "source_project": str(project),
                "judgments": (
                    splits
                    if splits is not None
                    else [{"name": "development", "path": judged}]
                ),
                "harness": harness or {},
                "arms": arms,
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(
    spec_path: Path,
    stub_engine: Path,
    workspace: Path,
    runs: Path,
    *,
    validate_first: bool = True,
    keep_sandboxes: bool = True,
    behaviour: str | None = None,
):
    """Drive a real run, with the stub's knobs held in this process's environment."""

    if behaviour is not None:
        os.environ["RAG_EXPERIMENTS_STUB_BEHAVIOUR"] = behaviour
    try:
        return run_experiment(
            load_spec(spec_path),
            app_source=locate_engine(stub_engine),
            workspace=workspace,
            runs_directory=runs,
            keep_sandboxes=keep_sandboxes,
            validate_first=validate_first,
        )
    finally:
        os.environ.pop("RAG_EXPERIMENTS_STUB_BEHAVIOUR", None)


@pytest.fixture(autouse=True)
def _stub_knows_the_original(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_EXPERIMENTS_STUB_ORIGINAL", str(project))


def test_a_run_measures_every_arm_and_writes_one_record(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[
            {"name": "baseline", "kind": "settings", "overlay": {}},
            {
                "name": "deeper",
                "kind": "settings",
                "overlay": {"retrieval.maximum_candidates": 80},
            },
        ],
        harness={"modes": ["bm25", "hybrid+rerank"], "top_k": 10},
    )
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    record = outcome.record

    assert outcome.verdict == VERDICT_VERIFIED, outcome.comparison
    assert outcome.error is None
    assert outcome.guard_state == "unchanged"
    assert len(record["arms"]) == 2
    assert [arm["arm"]["name"] for arm in record["arms"]] == ["baseline", "deeper"]
    for arm in record["arms"]:
        assert arm["measured"] is True
        assert arm["measured_splits"] == ["development"]
        assert arm["sandbox"]["project_id"] == PROJECT_ID
        assert arm["sandbox"]["generation_ids"] == [FIRST_GENERATION]
        # The sandbox the harness was given is a copy, never the project.
        assert arm["sandbox"]["root"] != str(project)
        assert arm["sandbox"]["source_root"] == str(
            Path(arm["sandbox"]["root"]) / "sources"
        )
        assert arm["settings"]["key_count"] > 30
        assert arm["settings"]["document_sha256"]
        assert Path(arm["settings"]["model_cache_root"]).is_absolute()

    # The pinned file is the one the app's own reader uses, and it states the arm's
    # overlay.
    deeper = record["arms"][1]
    pinned = Path(deeper["sandbox"]["settings_file"])
    assert "maximum_candidates = 80" in pinned.read_text(encoding="utf-8")
    baseline = Path(record["arms"][0]["sandbox"]["settings_file"])
    assert "maximum_candidates = 80" not in baseline.read_text(encoding="utf-8")

    # The judged bytes are kept under the run, with the digest of what was measured.
    judged = record["judgments"][0]
    assert Path(judged["kept_at"]).is_file()
    assert judged["kept_at"].startswith(str(outcome.run_directory))
    assert judged["sha256"] == judged["kept_sha256"]
    assert judged["sha256"] == _digest(Path(judged["path"]))

    # The engine is named twice: once as it was, once as it is.
    assert record["engine"]["revision"] is None  # the stub tree is not a checkout
    assert record["engine"]["content_sha256_after"]
    assert record["engine"]["changed"] is False


def test_every_byte_a_run_produced_is_inside_its_own_directories(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    run_directory = outcome.run_directory
    assert outcome.run_directory.parent == tmp_path / "runs"

    arm = outcome.record["arms"][0]
    assert Path(arm["log"]).parent == run_directory / "baseline"
    assert Path(arm["log"]).is_file()
    assert Path(arm["measure"][0]["report"]).parent == run_directory / "baseline"
    assert outcome.record_path == run_directory / "run.json"
    assert (run_directory / "judged" / "development.json").is_file()

    # The log holds the harness's own output, so a reader who cannot rerun the
    # measurement still has what it printed.
    log = Path(arm["log"]).read_text(encoding="utf-8")
    assert "harness: 2 searches" in log
    assert "harness: pinned settings at" in log
    assert "--- stdout ---" in log


def test_two_runs_of_one_specification_keep_their_own_evidence(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    first = _run(spec, stub_engine, workspace, tmp_path / "runs")
    first_evidence = {
        path.relative_to(first.run_directory): _digest(path)
        for path in sorted(first.run_directory.rglob("*"))
        if path.is_file()
    }
    second = _run(spec, stub_engine, workspace, tmp_path / "runs")

    assert first.run_directory != second.run_directory
    assert first.verdict == second.verdict == VERDICT_VERIFIED
    # The second run wrote nothing over the first run's reports, logs, or record.
    for relative, digest in first_evidence.items():
        assert _digest(first.run_directory / relative) == digest, relative
    # And each sandbox carries its own run's name, so neither collides with the other.
    assert (
        Path(second.record["arms"][0]["sandbox"]["workspace"]).name
        != Path(first.record["arms"][0]["sandbox"]["workspace"]).name
    )


def test_a_dry_run_then_a_real_run_both_succeed(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path, capsys
) -> None:
    from rag_experiments.cli import main

    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    arguments = [
        "run",
        "--spec",
        str(spec),
        "--app-source",
        str(stub_engine),
        "--workspace",
        str(workspace),
        "--runs",
        str(tmp_path / "runs"),
    ]
    assert main([*arguments, "--dry-run"]) == 0
    capsys.readouterr()
    assert not list((tmp_path / "runs").rglob("run.json"))

    assert main(arguments) == 0
    out = capsys.readouterr().out
    assert "record:" in out


def test_an_arm_that_refuses_still_leaves_a_record(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[
            {"name": "baseline", "kind": "settings", "overlay": {}},
            {"name": "deeper", "kind": "settings", "overlay": {}},
        ],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(
        spec,
        stub_engine,
        workspace,
        tmp_path / "runs",
        behaviour="measure-refuses",
    )
    record = outcome.record
    assert outcome.verdict == VERDICT_FAILED
    assert outcome.error is not None
    assert "reranker model not installed" in outcome.error
    assert record["run"]["complete"] is False
    # The arm that refused is in the record with the evidence it produced.
    arm = record["arms"][0]
    assert arm["measured"] is False
    assert arm["failure_stage"] == "measure (development)"
    assert "reranker model not installed" in arm["failure"]
    assert arm["measure"][0]["exit_code"] == 2
    assert "harness: reranker model not installed" in Path(arm["log"]).read_text(
        encoding="utf-8"
    )
    # Zero successes is still a record, and it never claims a measurement.
    assert record["measured_arm_count"] == 0
    assert Path(outcome.record_path).is_file()


def test_an_arm_that_fails_to_prepare_still_leaves_a_record(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[
            {"name": "baseline", "kind": "settings", "overlay": {"chunking.size": 256}},
            {
                "name": "bad",
                "kind": "settings",
                "overlay": {"retrieval.no_such_knob": 1},
            },
        ],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    record = outcome.record
    assert outcome.verdict == VERDICT_INCOMPLETE
    assert "retrieval.no_such_knob" in outcome.error
    assert len(record["arms"]) == 2
    assert record["measured_arm_count"] == 1
    assert record["arms"][0]["measured"] is True
    assert record["arms"][1]["failure_stage"] == "prepare"
    assert "retrieval.no_such_knob" in record["arms"][1]["failure"]


def test_a_run_that_measured_nothing_and_refused_says_failed(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[
            {
                "name": "bad",
                "kind": "settings",
                "overlay": {"retrieval.no_such_knob": 1},
            }
        ],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert outcome.verdict == VERDICT_FAILED
    assert outcome.record["measured_arm_count"] == 0


def test_a_guard_that_cannot_be_closed_says_unknown_and_keeps_the_record(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path, monkeypatch
) -> None:
    # The project becoming unreadable, or the app's own names changing under the
    # run, must not cost the record of what was measured.
    from rag_experiments.report import record as record_module

    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    real = record_module.snapshot
    seen = {"calls": 0}

    def _refusing(root: Path):
        # The opening digest succeeds; the closing one cannot be taken, which is the
        # case the record has to hold rather than lose.
        seen["calls"] += 1
        if seen["calls"] == 2:
            raise OSError("the project went away mid-run")
        return real(root)

    monkeypatch.setattr(record_module, "snapshot", _refusing)
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert outcome.guard_state == "unknown"
    assert outcome.verdict == VERDICT_FAILED
    guard = outcome.record["source_project"]["guard"]
    assert guard["state"] == "unknown"
    assert "went away" in guard["error"]
    assert guard["before"]["digest"]
    assert guard["after"] is None
    # The arms did measure, and their reports are still in the record.
    assert outcome.record["measured_arm_count"] == 1
    assert Path(outcome.record["arms"][0]["measure"][0]["report"]).is_file()


def test_a_source_project_that_moved_during_the_run_disqualifies_it(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    # The prepare command writes into the original. Nothing stops a patched engine
    # from doing the same, which is why the guard is around the whole run rather
    # than around the harness.
    victim = tmp_path / "elsewhere" / "a-book.pdf"
    spec = _spec_file(
        tmp_path,
        project,
        arms=[
            {
                "name": "baseline",
                "kind": "settings",
                "overlay": {},
                "prepare": [["sh", "-c", f"printf x >> {victim}"]],
            }
        ],
        harness={"modes": ["bm25"]},
    )
    original = project / "sources" / "a-book.pdf"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.symlink_to(original)
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert outcome.verdict == VERDICT_SOURCE_CHANGED
    assert outcome.guard_state == "changed"
    differences = outcome.record["source_project"]["guard"]["differences"]
    assert "changed: sources/a-book.pdf" in differences
    # The arms did run, and their numbers are disqualified rather than hidden.
    assert outcome.record["measured_arm_count"] == 1


def test_a_corpus_edited_while_the_run_goes_disqualifies_it(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    before = _digest(project / "sources" / "a-book.pdf")
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert outcome.verdict == VERDICT_VERIFIED
    assert before == _digest(project / "sources" / "a-book.pdf")


def test_a_validation_failure_stops_the_run_before_it_searches(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(
        spec,
        stub_engine,
        workspace,
        tmp_path / "runs",
        behaviour="validate-refuses",
    )
    assert outcome.verdict == VERDICT_FAILED
    arm = outcome.record["arms"][0]
    assert arm["failure_stage"] == "validate (development)"
    assert arm["measure"] == []
    assert "resolved to 0 chunks" in arm["failure"]


def test_an_interrupted_run_leaves_a_record_and_re_raises(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path, capsys
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    with pytest.raises(KeyboardInterrupt):
        _run(
            spec,
            stub_engine,
            workspace,
            tmp_path / "runs",
            behaviour="interrupted",
            validate_first=False,
        )
    records = list((tmp_path / "runs").rglob("run.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["verdict"] == VERDICT_INTERRUPTED
    assert record["interruption"].startswith("KeyboardInterrupt")
    # The guard was closed even though the run did not finish.
    assert record["source_project"]["guard"]["state"] == "unchanged"
    # The arm that was interrupted is in the record with what it had done: the log
    # it was writing, and the stage the interrupt arrived in.
    arm = record["arms"][0]
    assert arm["measured"] is False
    assert arm["failure_stage"] == "interrupted"
    assert arm["failure"].startswith("KeyboardInterrupt")
    assert Path(arm["log"]).is_file()
    assert "INTERRUPTED" in Path(arm["log"]).read_text(encoding="utf-8")


def test_an_engine_that_moved_while_the_arms_ran_disqualifies_them(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path, monkeypatch
) -> None:
    from rag_experiments import run as run_module

    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    # A code change during the run means the arms measured two engines. The digest
    # is the same one the record states, so this drives the real mechanism rather
    # than writing the field by hand.
    from rag_experiments.engine import locate

    real = locate.tree_digest
    seen = {"calls": 0}

    def _moving(root: Path):
        seen["calls"] += 1
        return "a" * 64 if seen["calls"] == 1 else "b" * 64

    monkeypatch.setattr(locate, "tree_digest", _moving)
    monkeypatch.setattr(run_module, "tree_digest", _moving)
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert seen["calls"] >= 2, "the engine is digested before and after the arms"
    assert outcome.verdict == VERDICT_ENGINE_CHANGED
    assert outcome.record["engine"]["content_sha256"] == "a" * 64
    assert outcome.record["engine"]["content_sha256_after"] == "b" * 64
    assert outcome.record["engine"]["changed"] is True
    assert real is not _moving


def test_a_report_that_is_not_the_measurement_refuses_the_arm(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    for behaviour, expected in (
        ("malformed-report", "not readable JSON"),
        ("no-report", "wrote no report"),
        ("wrong-depth", "different depth"),
    ):
        (tmp_path / behaviour).mkdir(exist_ok=True)
        spec = _spec_file(
            tmp_path / behaviour,
            project,
            arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
            harness={"modes": ["bm25"], "top_k": 10},
        )
        outcome = _run(
            spec,
            stub_engine,
            workspace / behaviour,
            tmp_path / behaviour / "runs",
            behaviour=behaviour,
        )
        assert outcome.verdict == VERDICT_FAILED, behaviour
        arm = outcome.record["arms"][0]
        assert arm["failure_stage"] == "report", behaviour
        assert expected in arm["failure"], (behaviour, arm["failure"])


def test_two_splits_are_each_measured_from_the_file_this_run_kept(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    for name in ("development.json", "held-out.json"):
        build_judged_set(tmp_path / name)
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
        splits=[
            {"name": "development", "path": "development.json"},
            {"name": "held-out", "path": "held-out.json"},
        ],
    )
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert outcome.verdict == VERDICT_VERIFIED
    arm = outcome.record["arms"][0]
    assert arm["measured_splits"] == ["development", "held-out"]
    assert [item["split"] for item in arm["measure"]] == ["development", "held-out"]
    assert arm["validate"] and len(arm["validate"]) == 2
    for split in ("development", "held-out"):
        report = Path(arm["reports"][split])
        assert report.name == f"report-{split}.json"
        measured = json.loads(report.read_text(encoding="utf-8"))
        assert measured["judgments"]["path"].endswith(f"{split}.json")
        # The bytes the measurement read are the bytes kept under the run.
        kept = Path(
            next(
                item["kept_at"]
                for item in outcome.record["judgments"]
                if item["name"] == split
            )
        )
        assert _digest(kept) == _digest(tmp_path / f"{split}.json")
    # Each measurement is checked against the file it claims to have read, and each
    # reports the query rows it ran under its own query ids.
    assert {item["report_facts"]["run_count"] for item in arm["measure"]} == {2}
    for item in arm["measure"]:
        rows = item["report_facts"]["by_mode"]["bm25"]
        assert list(rows["query_ids"]) == ["q01", "q02"]
        assert rows["summary_count"] == 2


def test_a_prepare_command_sees_the_copies_paths_and_not_the_originals(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[
            {
                "name": "baseline",
                "kind": "settings",
                "overlay": {},
                "prepare": [
                    [
                        "sh",
                        "-c",
                        (
                            "printf '%s\\n' \"$PWD\" > {sandbox}/where.txt; "
                            "printf '%s\\n' \"{source}\" > {sandbox}/source.txt; "
                            "printf '%s\\n' \"{tree}\" > {sandbox}/tree.txt"
                        ),
                    ]
                ],
            }
        ],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert outcome.verdict == VERDICT_VERIFIED
    sandbox = Path(outcome.record["arms"][0]["sandbox"]["workspace"])
    assert (sandbox / "where.txt").read_text(encoding="utf-8").strip() == str(
        Path(outcome.record["arms"][0]["sandbox"]["root"])
    )
    # `{source}` is the copy's own originals: a command that wrote there would
    # rewrite a sandbox, not the corpus.
    assert (sandbox / "source.txt").read_text(encoding="utf-8").strip() == str(
        Path(outcome.record["arms"][0]["sandbox"]["source_root"])
    )
    assert (sandbox / "tree.txt").read_text(encoding="utf-8").strip() == str(
        stub_engine
    )


def test_a_run_refuses_a_workspace_inside_the_project_it_measures(
    project: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    inside = project / "runs"
    inside.mkdir()
    with pytest.raises(ExperimentError) as caught:
        run_experiment(
            load_spec(spec),
            app_source=locate_engine(stub_engine),
            workspace=inside,
            runs_directory=inside,
        )
    assert "inside the source project" in str(caught.value)


def test_a_run_refuses_evidence_directories_inside_the_engine(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    inside = stub_engine / "runs"
    inside.mkdir()
    try:
        spec = _spec_file(
            tmp_path,
            project,
            arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
            harness={"modes": ["bm25"]},
        )
        with pytest.raises(ExperimentError) as caught:
            run_experiment(
                load_spec(spec),
                app_source=locate_engine(stub_engine),
                workspace=workspace,
                runs_directory=inside,
            )
        assert "inside the engine tree" in str(caught.value)
    finally:
        inside.rmdir()


def test_a_run_that_measured_nothing_with_no_error_says_so(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    from rag_experiments.experiment.spec import RunSpec

    # A specification with no arms cannot be loaded, so this is the shape a run
    # reaches when every arm is removed between loading and measuring. The record is
    # still written and says no arm measured anything rather than claiming success.
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    loaded = load_spec(spec)
    empty = RunSpec(
        path=loaded.path,
        name=loaded.name,
        source_project=loaded.source_project,
        judgments=loaded.judgments,
        generations=loaded.generations,
        harness=loaded.harness,
        arms=(),
    )
    outcome = run_experiment(
        empty,
        app_source=locate_engine(stub_engine),
        workspace=workspace,
        runs_directory=tmp_path / "runs",
    )
    assert outcome.verdict == VERDICT_NO_ARMS
    assert outcome.record["arms"] == []
    assert outcome.record_path.is_file()


def test_a_sandbox_can_be_removed_after_a_run_and_the_evidence_stays(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    from rag_experiments.sandbox import remove

    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(
        spec, stub_engine, workspace, tmp_path / "runs", keep_sandboxes=False
    )
    assert outcome.verdict == VERDICT_VERIFIED
    arm = outcome.record["arms"][0]
    assert arm["sandbox_removed"] is True
    assert not Path(arm["sandbox"]["root"]).exists()
    # Everything a reader needs is in the run's own directory and is untouched.
    assert Path(arm["log"]).is_file()
    assert Path(arm["measure"][0]["report"]).is_file()
    written = Path(arm["settings"]["written_to"])
    assert written.is_relative_to(Path(arm["sandbox"]["root"]))
    assert not written.exists()
    # And a `remove` by hand has nothing left to do, because the run already did it.
    with pytest.raises(ExperimentError):
        remove(workspace, Path(arm["sandbox"]["workspace"]).name)


def test_the_engine_under_test_is_the_one_the_run_names(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    engine = outcome.record["engine"]
    assert engine["root"] == str(stub_engine)
    assert engine["harness"] == str(stub_engine / "scripts" / "evaluate_retrieval.py")
    arm = outcome.record["arms"][0]
    assert arm["engine"]["root"] == str(stub_engine)
    # The child was told which tree to import from, and told not to write bytecode
    # into it.
    assert arm["environment"]["given"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert arm["environment"]["given"]["PYTHONPATH"] == str(stub_engine / "src")
    assert not (stub_engine / "src" / "research_rag" / "__pycache__").exists()


def test_a_run_never_writes_into_the_engine_tree(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    def tree_state() -> dict[str, str]:
        return {
            str(path.relative_to(stub_engine)): _digest(path)
            for path in sorted(stub_engine.rglob("*"))
            if path.is_file()
        }

    before = tree_state()
    spec = _spec_file(
        tmp_path,
        project,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert tree_state() == before


def test_a_code_arm_measures_a_clone_of_the_tree_under_test(
    project: Path, workspace: Path, tmp_path: Path, git_engine: Path
) -> None:
    from rag_experiments.engine.locate import git_revision_of

    patch = tmp_path / "variant.patch"
    patch.write_text(
        "diff --git a/src/research_rag/variant.py b/src/research_rag/variant.py\n"
        "new file mode 100644\n"
        "index 0000000..8b13789\n"
        "--- /dev/null\n"
        "+++ b/src/research_rag/variant.py\n"
        "@@ -0,0 +1 @@\n"
        '+VARIANT = "measured"\n',
        encoding="utf-8",
    )
    spec = _spec_file(
        tmp_path,
        project,
        arms=[
            {
                "name": "patched",
                "kind": "code",
                "base": git_revision_of(git_engine),
                "patch": str(patch),
            }
        ],
        harness={"modes": ["bm25"]},
    )
    outcome = run_experiment(
        load_spec(spec),
        app_source=locate_engine(git_engine),
        workspace=workspace,
        runs_directory=tmp_path / "runs",
    )
    record = outcome.record
    assert outcome.verdict == VERDICT_VERIFIED, outcome.error
    checkout = record["arms"][0]["checkout"]
    # The tree the run measured is a clone, and the patch landed in the clone and
    # nowhere else: the working tree the arm was measured against never changed.
    assert checkout["method"].startswith("git clone")
    assert Path(checkout["root"]) != git_engine
    assert Path(checkout["root"], "src", "research_rag", "variant.py").is_file()
    assert not (git_engine / "src" / "research_rag" / "variant.py").exists()
    # The arm measured the clone, not the tree it was cloned from.
    assert record["arms"][0]["engine"]["root"] == checkout["root"]
    assert record["arms"][0]["measured"] is True
    # The patch and what the tree it copied held are kept inside the run.
    assert Path(checkout["patch"]).is_relative_to(outcome.run_directory)
    assert Path(checkout["evidence_directory"], "engine-source.json").is_file()
    assert (outcome.run_directory / "patched" / "arm.log").is_file()


def _digest(path: Path) -> str:
    accumulator = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            accumulator.update(block)
    return accumulator.hexdigest()


def test_a_run_against_a_project_that_is_not_one_is_refused(
    tmp_path: Path, workspace: Path, stub_engine: Path
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    spec = _spec_file(
        tmp_path,
        plain,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert outcome.verdict == VERDICT_FAILED
    assert "not a research-rag project" in outcome.error
    assert outcome.record_path.is_file()


def test_the_run_process_reports_a_refusal_without_a_traceback(
    project: Path, workspace: Path, tmp_path: Path, stub_engine: Path
) -> None:
    # The same run through the command line, as a reader would meet it: a non-zero
    # status, the reason on stderr, and the record's path on stdout.
    spec = _spec_file(
        tmp_path,
        project,
        arms=[
            {
                "name": "bad",
                "kind": "settings",
                "overlay": {"retrieval.no_such_knob": 1},
            }
        ],
        harness={"modes": ["bm25"]},
    )
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            (
                "import sys; from rag_experiments.cli import main; "
                "sys.exit(main(sys.argv[1:]))"
            ),
            "run",
            "--spec",
            str(spec),
            "--app-source",
            str(stub_engine),
            "--workspace",
            str(workspace),
            "--runs",
            str(tmp_path / "runs"),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "RAG_EXPERIMENTS_STUB_ORIGINAL": str(project)},
        check=False,
    )
    assert completed.returncode == 1
    assert "retrieval.no_such_knob" in completed.stderr
    assert "record:" in completed.stdout
    record = next((tmp_path / "runs").rglob("run.json"))
    assert json.loads(record.read_text(encoding="utf-8"))["verdict"] == VERDICT_FAILED


def test_a_project_naming_a_custom_source_directory_is_measured_from_it(
    tmp_path: Path, workspace: Path, stub_engine: Path
) -> None:
    root = build_project(tmp_path / "corpus")
    (root / "sources").rename(root / "reading-list")
    descriptor = root / ".research-rag" / "project.json"
    document = json.loads(descriptor.read_text(encoding="utf-8"))
    document["source_directory"] = "reading-list"
    descriptor.write_text(json.dumps(document), encoding="utf-8")

    spec = _spec_file(
        tmp_path,
        root,
        arms=[{"name": "baseline", "kind": "settings", "overlay": {}}],
        harness={"modes": ["bm25"]},
    )
    outcome = _run(spec, stub_engine, workspace, tmp_path / "runs")
    assert outcome.verdict == VERDICT_VERIFIED
    assert outcome.record["arms"][0]["sandbox"]["source_root"] == str(
        Path(outcome.record["arms"][0]["sandbox"]["root"]) / "reading-list"
    )
