"""That an arm's command says what the specification asked for.

The measurement is the app's own harness, run as a subprocess. These tests hold
the part this harness owns: which flags reach that harness, which settings reach
the pinned file, what a code arm's checkout is made of, and that a report which is
not the measurement that was asked for refuses the arm instead of being printed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from conftest import FIRST_GENERATION, PROJECT_ID, build_judged_set

from rag_experiments.engine.harness import HarnessContract, harness_contract
from rag_experiments.engine.locate import git_revision_of, locate_engine
from rag_experiments.errors import ExperimentError
from rag_experiments.experiment.checkout import (
    CLONE_METHOD,
    PATCH_COPY,
    SOURCE_DIFF,
    SOURCE_EVIDENCE,
    make_checkout,
)
from rag_experiments.experiment.execute import (
    HARNESS_JOINED,
    HARNESS_REPEATED,
    HARNESS_SETTINGS,
    VALIDATE_FLAG,
    _harness_flag,
    _interpolate,
    effective_overlay,
    prepare_arm,
    requested_modes,
)
from rag_experiments.experiment.report_schema import (
    QuerySelection,
    ReportFacts,
    ReportValidationError,
    select_queries,
    validate_report,
)
from rag_experiments.experiment.spec import Arm, JudgedSet, RunSpec, segment


def _spec(**overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "path": Path("/tmp/spec.json"),
        "name": "fixture",
        "source_project": Path("/tmp/corpus"),
        "judgments": (),
        "generations": (),
        "harness": {},
        "arms": (),
    }
    base.update(overrides)
    return RunSpec(**base)  # type: ignore[arg-type]


def _arm(**overrides: object) -> Arm:
    base: dict[str, object] = {
        "name": "baseline",
        "kind": "settings",
        "overlay": {},
        "base": None,
        "patch": None,
        "prepare": (),
    }
    base.update(overrides)
    return Arm(**base)  # type: ignore[arg-type]


def _prepared(spec: RunSpec, arm: Arm, workspace: Path, app_source: Path):
    return prepare_arm(
        spec,
        arm,
        workspace=workspace,
        sandbox_name=f"run-{segment(arm.name)}",
        arm_directory=workspace / "run" / segment(arm.name),
        app_source=locate_engine(app_source),
    )


def test_a_joined_key_becomes_one_comma_separated_value() -> None:
    # The app's --modes is a single value. Passing it twice measures the last one,
    # so a list here is joined rather than repeated.
    assert _harness_flag("modes", ["bm25", "dense", "hybrid"]) == [
        "--modes",
        "bm25,dense,hybrid",
    ]
    assert "modes" in HARNESS_JOINED


def test_a_repeated_key_becomes_one_flag_per_value() -> None:
    assert _harness_flag("reranker_model", ["a/model", "b/model"]) == [
        "--reranker-model",
        "a/model",
        "--reranker-model",
        "b/model",
    ]
    assert "reranker_model" in HARNESS_REPEATED


def test_a_boolean_key_becomes_a_stored_flag_or_nothing() -> None:
    assert _harness_flag("offline", True) == ["--offline"]
    assert _harness_flag("offline", False) == []


def test_a_scalar_key_becomes_one_flag_and_its_value() -> None:
    assert _harness_flag("top_k", 10) == ["--top-k", "10"]
    assert _harness_flag("deep_top_k", 0) == ["--deep-top-k", "0"]


def test_a_key_with_no_value_contributes_nothing() -> None:
    assert _harness_flag("limit", None) == []


def test_a_list_on_a_key_that_takes_one_value_is_refused() -> None:
    with pytest.raises(ExperimentError) as caught:
        _harness_flag("top_k", [10, 20])
    assert "top_k" in str(caught.value)
    assert "reranker_model" in str(caught.value)


def test_an_empty_list_contributes_nothing() -> None:
    assert _harness_flag("modes", []) == []
    assert _harness_flag("reranker_model", []) == []


def test_a_harness_entry_that_is_a_setting_reaches_the_overlay() -> None:
    overlay = effective_overlay(
        _arm(overlay={"chunking.size": 512}),
        {"offline": True, "dense_backend": "exact"},
    )
    assert overlay == {
        "runtime.offline": True,
        "dense.backend": "exact",
        "chunking.size": 512,
    }
    assert HARNESS_SETTINGS["offline"] == "runtime.offline"


def test_the_arms_own_overlay_wins_over_the_shared_harness_block() -> None:
    # The `harness` block is what every arm of the run shares; the overlay is the
    # arm-specific statement, so it decides.
    overlay = effective_overlay(
        _arm(overlay={"dense.backend": "qdrant"}), {"dense_backend": "exact"}
    )
    assert overlay == {"dense.backend": "qdrant"}


def test_a_harness_entry_that_is_not_a_setting_stays_out_of_the_overlay() -> None:
    overlay = effective_overlay(_arm(), {"top_k": 10, "modes": ["hybrid"]})
    assert overlay == {}


def test_the_modes_a_run_asked_for_are_the_ones_it_named(
    app_source: Path,
) -> None:
    contract = harness_contract(locate_engine(app_source))
    named = _spec(harness={"modes": ["bm25", "hybrid"]})
    assert requested_modes(named, contract) == ("bm25", "hybrid")
    assert requested_modes(_spec(harness={"modes": "bm25, dense"}), contract) == (
        "bm25",
        "dense",
    )
    # A specification that names none is asking for everything the engine's harness
    # declares, which is read from the engine rather than guessed here.
    assert requested_modes(_spec(), contract) == contract.modes
    with pytest.raises(ExperimentError) as caught:
        requested_modes(_spec())
    assert "no `modes`" in str(caught.value)


class _StubSandbox:
    """The three attributes `_interpolate` reads, so it can be tested alone."""

    def __init__(self, root: Path) -> None:
        self.root = root / "project"
        self.source_root = root / "project" / "sources"
        self.workspace = root / "sandbox"


def test_every_harness_token_expands_and_an_unknown_one_is_left_alone(
    app_source: Path, tmp_path: Path
) -> None:
    sandbox = _StubSandbox(tmp_path)
    engine = locate_engine(app_source)
    arguments = _interpolate(
        "cmd {project} {source} {tree} {sandbox} {unknown}",
        sandbox,  # type: ignore[arg-type]
        engine,
    )
    assert arguments == [
        "cmd",
        str(tmp_path / "project"),
        str(tmp_path / "project" / "sources"),
        str(engine.root),
        str(tmp_path / "sandbox"),
        "{unknown}",
    ]


def test_a_prepare_command_is_split_the_way_a_shell_would(app_source: Path) -> None:
    arguments = _interpolate(
        "echo 'a b' c",
        _StubSandbox(Path("/tmp")),  # type: ignore[arg-type]
        locate_engine(app_source),
    )
    assert arguments == ["echo", "a b", "c"]


def test_preparing_an_arm_pins_every_setting_and_keeps_the_id(
    project: Path, workspace: Path, app_source: Path, tmp_path: Path
) -> None:
    judged = tmp_path / "judged.json"
    build_judged_set(judged)
    spec = _spec(
        name="fixture",
        source_project=project,
        judgments=(JudgedSet(name="all", path=judged),),
        harness={"top_k": 10, "offline": True},
        arms=(_arm(overlay={"retrieval.rrf_k": 120}),),
    )
    prepared = _prepared(spec, spec.arms[0], workspace, app_source)
    document = (prepared.sandbox.settings_file).read_text(encoding="utf-8")
    assert "rrf_k = 120" in document
    assert "offline = true" in document, "the harness block is a setting too"
    assert prepared.settings["key_count"] > 30
    assert sorted(prepared.settings["overridden"]) == [
        "retrieval.rrf_k",
        "runtime.offline",
    ]
    assert prepared.settings["document_sha256"]
    assert prepared.sandbox.generation_ids == (FIRST_GENERATION,)
    assert prepared.engine.root == app_source
    # The model cache the sandbox will read is pinned as an absolute path, so the
    # copy does not depend on an inherited configuration or cache home.
    assert Path(prepared.settings["model_cache_root"]).is_absolute()
    assert prepared.settings["config_home"] == str(prepared.sandbox.config_home)
    assert prepared.log_path.is_file()
    assert prepared.log_path.parent == workspace / "run" / "baseline"


def test_a_code_arm_that_changes_nothing_is_refused_by_the_specification(
    make_spec,
) -> None:
    from rag_experiments.experiment.spec import load_spec

    with pytest.raises(ExperimentError):
        load_spec(make_spec([{"name": "a", "kind": "code"}]))


def test_a_settings_arm_is_measured_by_the_tree_under_test(
    project: Path, workspace: Path, app_source: Path
) -> None:
    spec = _spec(name="fixture", source_project=project, arms=(_arm(),))
    prepared = _prepared(spec, spec.arms[0], workspace, app_source)
    # A settings arm runs the engine as it stands, with no checkout of its own.
    assert prepared.checkout is None
    assert prepared.engine.root == app_source
    assert prepared.settings["overridden"] == []


def test_a_sandbox_inside_the_project_it_measures_is_refused(
    project: Path, app_source: Path
) -> None:
    spec = _spec(name="fixture", source_project=project, arms=(_arm(),))
    inside = project / "workspaces"
    inside.mkdir()
    with pytest.raises(ExperimentError) as caught:
        prepare_arm(
            spec,
            spec.arms[0],
            workspace=inside,
            sandbox_name="run-baseline",
            arm_directory=inside / "run" / "baseline",
            app_source=locate_engine(app_source),
        )
    assert "inside the source project" in str(caught.value)


def test_the_record_form_of_an_arm_names_its_kind_and_overlay() -> None:
    arm = _arm(overlay={"chunking.size": 512})
    described = arm.describe()
    assert described == {
        "name": "baseline",
        "kind": "settings",
        "overlay": {"chunking.size": 512},
        "base": None,
        "patch": None,
        "prepare": [],
    }


def test_a_prepared_arm_record_is_json_serialisable_whole(
    project: Path, workspace: Path, app_source: Path
) -> None:
    spec = _spec(name="fixture", source_project=project, arms=(_arm(),))
    prepared = _prepared(spec, spec.arms[0], workspace, app_source)
    # A record the reader cannot load is a record nobody can act on.
    described = json.loads(json.dumps(prepared.describe()))
    assert described["sandbox"]["project_id"] == PROJECT_ID
    assert described["sandbox"]["source_root"] == str(prepared.sandbox.source_root)


def _two_split_spec(project: Path, tmp_path: Path) -> RunSpec:
    for name in ("development.json", "held-out.json"):
        build_judged_set(tmp_path / name)
    return _spec(
        name="fixture",
        source_project=project,
        judgments=(
            JudgedSet(name="development", path=tmp_path / "development.json"),
            JudgedSet(name="held-out", path=tmp_path / "held-out.json"),
        ),
        arms=(_arm(),),
    )


def test_the_measurement_command_carries_the_split_the_app_harness_expects(
    app_source: Path, project: Path, tmp_path: Path
) -> None:
    from rag_experiments.experiment.execute import _measure_command

    spec = _two_split_spec(project, tmp_path)
    engine = locate_engine(app_source)
    sandbox = _StubSandbox(tmp_path)  # type: ignore[arg-type]
    command = _measure_command(
        spec,
        engine,
        sandbox,  # type: ignore[arg-type]
        spec.judgments[1],
        tmp_path / "report.json",
        stage="measure",
    )
    assert str(spec.judgments[1].path) in command
    assert str(spec.judgments[0].path) not in command
    assert "--validate-only" not in command
    assert command[1] == str(engine.harness)


def test_the_validation_command_carries_the_apps_own_flag(
    app_source: Path, project: Path, tmp_path: Path
) -> None:
    from rag_experiments.experiment.execute import _measure_command

    spec = _two_split_spec(project, tmp_path)
    command = _measure_command(
        spec,
        locate_engine(app_source),
        _StubSandbox(tmp_path),  # type: ignore[arg-type]
        spec.judgments[0],
        tmp_path / "report.json",
        stage="validate",
    )
    assert command[-1] == VALIDATE_FLAG


def test_each_split_gets_its_own_report_file(app_source: Path, tmp_path: Path) -> None:
    from rag_experiments.experiment.execute import REPORT_PATTERN

    assert REPORT_PATTERN.format(split="held-out") == "report-held-out.json"
    # A split name is a record key and a printed label, so it is refused unless it
    # is a safe file name rather than quietly renamed.
    assert segment("held out/query", strict=False) == "held-out-query"
    with pytest.raises(ExperimentError):
        segment("held out/query")


def test_a_split_name_that_is_not_a_path_stays_one_segment() -> None:
    assert segment("../../etc/passwd", strict=False) == "etc-passwd"
    assert segment("///", strict=False) == "split"


def test_a_measured_arm_reports_one_entry_per_split(
    project: Path, workspace: Path, app_source: Path, tmp_path: Path
) -> None:
    spec = _two_split_spec(project, tmp_path)
    prepared = _prepared(spec, _arm(), workspace, app_source)
    # Preparation is split-agnostic: one sandbox, one pinned file, one engine, and
    # the splits differ only in which judged set each measurement is handed.
    assert prepared.sandbox.generation_ids == (FIRST_GENERATION,)
    assert prepared.settings["key_count"] > 30
    # Each split's query set is computed while the arm is prepared, so a selection
    # the file cannot yield is refused before a search is run.
    assert sorted(prepared.selections) == ["development", "held-out"]
    for name, selection in prepared.selections.items():
        assert selection.query_ids == ("q01", "q02")
        assert selection.total_query_count == 2
        assert prepared.describe()["query_selection"][name]["query_count"] == 2


# --- the report a measurement is checked against ------------------------------


def _contract(app_source: Path) -> HarnessContract:
    return harness_contract(locate_engine(app_source))


def _judged(tmp_path: Path, *, queries: int = 2) -> tuple[Path, QuerySelection]:
    """A judged set with real query identities, and the selection it yields."""

    path = tmp_path / "judged.json"
    build_judged_set(path, queries=queries)
    return path, select_queries(path)


def _rows(
    modes: tuple[str, ...],
    query_ids: tuple[str, ...],
    *,
    top_k: int | None,
    reranked: bool = True,
    fallback: dict[str, str] | None = None,
    candidate_count: int = 3,
    deep_top_k: int | None = None,
    drop: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    """One row per query per mode, the way the engine's harness writes them."""

    rows: list[dict[str, object]] = []
    for query_id in query_ids:
        if query_id in drop:
            continue
        for mode in modes:
            rows.append(
                {
                    "query_id": query_id,
                    "mode": mode,
                    "top_k": deep_top_k if deep_top_k is not None else top_k,
                    "rank": 1,
                    "success_at_k": True,
                    "reciprocal_rank": 1.0,
                    "ndcg_at_k": 1.0,
                    "result_count": candidate_count,
                    "returned_chunk_ids": ["c01"],
                    "candidate_count": candidate_count,
                    "candidate_depth": 200,
                    "rerank_window": 60 if mode.startswith("hybrid+rerank") else 0,
                    "rerank_requested": mode.startswith("hybrid+rerank"),
                    "reranked": reranked,
                    "rerank_fallback": fallback,
                }
            )
    return rows


def _summary(
    modes: tuple[str, ...], rows: list[dict[str, object]], top_k: int | None
) -> dict[str, object]:
    return {
        mode: {
            "overall": {
                "query_count": sum(1 for row in rows if row["mode"] == mode),
                "top_k": top_k,
                "mrr": 1.0,
                "ndcg_at_k": 1.0,
                "p50_seconds": 0.5,
            }
        }
        for mode in modes
    }


def _report(
    tmp_path: Path,
    *,
    app_source: Path,
    selection: QuerySelection,
    modes: tuple[str, ...] = ("bm25",),
    top_k: int | None = 10,
    deep_top_k: int | None = None,
    judged_path: Path | None = None,
    judged_query_count: int | None = None,
    evaluated_query_count: int | None = None,
    classes: tuple[str, ...] | None = None,
    rows: list[dict[str, object]] | None = None,
    summary: dict[str, object] | None = None,
    schema_version: int | None = None,
    deep_rows: list[dict[str, object]] | None = None,
    name: str = "report.json",
) -> Path:
    """Write a report in the shape the engine's harness writes.

    The rows, the judged block, and the settings are all filled in from the
    selection this run asked for, so a test states only the one thing it is breaking:
    a fixture whose rows are made up would be testing a report no engine produces.
    """

    contract = _contract(app_source)
    written = (
        rows if rows is not None else _rows(modes, selection.query_ids, top_k=top_k)
    )
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "schema_version": (
                    contract.report_schema_version
                    if schema_version is None
                    else schema_version
                ),
                "judgments": {
                    "path": str(selection.path if judged_path is None else judged_path),
                    "schema_version": 1,
                    "query_count": (
                        selection.total_query_count
                        if judged_query_count is None
                        else judged_query_count
                    ),
                    "evaluated_query_count": (
                        len(selection.query_ids)
                        if evaluated_query_count is None
                        else evaluated_query_count
                    ),
                    "target_count": len(selection.query_ids),
                },
                "settings": {
                    "selected_modes": list(modes),
                    "selected_deep_modes": [] if not deep_top_k else list(modes),
                    "selected_classes": list(
                        selection.classes if classes is None else classes
                    ),
                    "skipped_targets": list(selection.skipped_targets),
                    "top_k": top_k,
                    "deep_top_k": deep_top_k if deep_top_k is not None else 0,
                    "allow_degraded_rerank": False,
                },
                "runs": written,
                "deep_runs": deep_rows or [],
                "summary": (
                    summary if summary is not None else _summary(modes, written, top_k)
                ),
                "deep_summary": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def _validate(
    path: Path,
    app_source: Path,
    *,
    selection: QuerySelection,
    modes: tuple[str, ...] = ("bm25",),
    top_k: int | None = 10,
    deep_top_k: int | None = None,
) -> ReportFacts:
    return validate_report(
        path,
        contract=_contract(app_source),
        requested_modes=modes,
        selection=selection,
        judged_sha256=_digest_of(selection.path),
        top_k=top_k,
        deep_top_k=deep_top_k,
        arm="baseline",
        split="development",
    )


def _digest_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refuse(path: Path, app_source: Path, selection: QuerySelection, **kwargs) -> str:
    """The message a refusal carries, for a report that must not be believed."""

    with pytest.raises(ReportValidationError) as caught:
        _validate(path, app_source, selection=selection, **kwargs)
    return str(caught.value)


def test_a_report_that_answers_for_the_run_is_accepted(
    tmp_path: Path, app_source: Path
) -> None:
    judged, selection = _judged(tmp_path)
    path = _report(tmp_path, app_source=app_source, selection=selection)
    facts = _validate(path, app_source, selection=selection)
    assert facts.schema_version == _contract(app_source).report_schema_version
    assert facts.top_k == 10
    assert facts.deep_top_k == 0, "a deep pass that was not run is a stated zero"
    assert facts.judged_path == str(judged)
    assert facts.judged_query_count == 2
    assert facts.evaluated_query_count == 2
    assert facts.run_count == 2
    assert facts.by_mode["bm25"].query_ids == selection.query_ids
    assert facts.by_mode["bm25"].summary_count == 2
    assert facts.sha256
    assert facts.describe()["by_mode"]["bm25"]["query_ids"] == list(selection.query_ids)


def test_a_disabled_deep_pass_stated_as_zero_is_not_a_missing_depth(
    tmp_path: Path, app_source: Path
) -> None:
    # `deep_top_k` of zero is the engine's way of saying the deep pass was not run.
    # Reading it as an absent depth would refuse a report that stated exactly what it
    # did, which is the defect this test holds.
    judged, selection = _judged(tmp_path)
    path = _report(tmp_path, app_source=app_source, selection=selection, deep_top_k=0)
    facts = _validate(path, app_source, selection=selection, deep_top_k=0)
    assert facts.deep_top_k == 0
    assert facts.deep_run_count == 0
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["settings"]["deep_top_k"] == 0
    assert document["deep_runs"] == []
    assert document["judgments"]["schema_version"] == 1
    assert str(judged) == document["judgments"]["path"]


def test_a_deep_pass_at_a_stated_depth_is_accepted(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(
        tmp_path,
        app_source=app_source,
        selection=selection,
        deep_top_k=50,
        deep_rows=_rows(("bm25",), selection.query_ids, top_k=None, deep_top_k=50),
    )
    facts = _validate(path, app_source, selection=selection, deep_top_k=50)
    assert facts.deep_top_k == 50
    assert facts.deep_run_count == 2
    assert facts.deep_top_ks == (50, 50)


def test_a_deep_pass_at_another_depth_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(
        tmp_path,
        app_source=app_source,
        selection=selection,
        deep_top_k=30,
        deep_rows=_rows(("bm25",), selection.query_ids, top_k=None, deep_top_k=20),
    )
    message = _refuse(path, app_source, selection, deep_top_k=30)
    assert "deep depth" in message or "deep_top_k=20" in message


def test_a_deep_pass_whose_rows_state_no_depth_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(
        tmp_path,
        app_source=app_source,
        selection=selection,
        deep_top_k=50,
        deep_rows=_rows(("bm25",), selection.query_ids, top_k=None, deep_top_k=50),
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    for row in document["deep_runs"]:
        del row["top_k"]
    path.write_text(json.dumps(document), encoding="utf-8")
    message = _refuse(path, app_source, selection, deep_top_k=50)
    assert "deep rows stating no top_k" in message
    assert "q01" in message


def test_a_report_that_is_not_json_is_refused(tmp_path: Path, app_source: Path) -> None:
    _, selection = _judged(tmp_path)
    path = tmp_path / "report.json"
    path.write_text("half a file", encoding="utf-8")
    assert "not readable JSON" in _refuse(path, app_source, selection)


def test_a_report_of_another_schema_version_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(
        tmp_path, app_source=app_source, selection=selection, schema_version=1
    )
    assert "different measurements" in _refuse(path, app_source, selection)


def test_a_report_missing_a_mode_that_was_asked_for_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(tmp_path, app_source=app_source, selection=selection)
    assert "dense" in _refuse(path, app_source, selection, modes=("bm25", "dense"))


def test_a_rerank_mode_named_with_a_model_answers_for_the_mode(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    labelled = "hybrid+rerank[bge-reranker-base]"
    path = _report(
        tmp_path, app_source=app_source, selection=selection, modes=(labelled,)
    )
    facts = _validate(path, app_source, selection=selection, modes=("hybrid+rerank",))
    assert facts.modes == (labelled,)


def test_a_report_measured_at_another_depth_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(tmp_path, app_source=app_source, selection=selection, top_k=50)
    assert "different depth" in _refuse(path, app_source, selection)


def test_a_row_measured_at_another_depth_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    # The stated depth can agree with the run while a row disagrees with it: the
    # summary would then be a mean over rows measured at two depths.
    _, selection = _judged(tmp_path)
    rows = _rows(("bm25",), selection.query_ids, top_k=10)
    rows[0]["top_k"] = 5
    path = _report(tmp_path, app_source=app_source, selection=selection, rows=rows)
    message = _refuse(path, app_source, selection)
    assert "q01" in message
    assert "top_k" in message


def test_a_report_stating_no_depth_is_refused(tmp_path: Path, app_source: Path) -> None:
    _, selection = _judged(tmp_path)
    path = _report(tmp_path, app_source=app_source, selection=selection)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["settings"]["top_k"]
    path.write_text(json.dumps(document), encoding="utf-8")
    assert "no top_k" in _refuse(path, app_source, selection)


def test_a_row_stating_no_depth_is_refused(tmp_path: Path, app_source: Path) -> None:
    _, selection = _judged(tmp_path)
    rows = _rows(("bm25",), selection.query_ids, top_k=10)
    del rows[1]["top_k"]
    path = _report(tmp_path, app_source=app_source, selection=selection, rows=rows)
    message = _refuse(path, app_source, selection)
    assert "q02" in message
    assert "no top_k" in message


def test_a_report_naming_no_judged_file_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(tmp_path, app_source=app_source, selection=selection)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["judgments"]["path"]
    path.write_text(json.dumps(document), encoding="utf-8")
    message = _refuse(path, app_source, selection)
    assert "naming no judged file" in message


def test_a_report_measured_from_another_judged_set_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    other = tmp_path / "other.json"
    build_judged_set(other)
    path = _report(
        tmp_path, app_source=app_source, selection=selection, judged_path=other
    )
    assert "Two judged sets" in _refuse(path, app_source, selection)


def test_a_report_stating_no_query_count_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(tmp_path, app_source=app_source, selection=selection)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["judgments"]["query_count"]
    path.write_text(json.dumps(document), encoding="utf-8")
    assert "no query count" in _refuse(path, app_source, selection)


def test_a_report_reading_another_number_of_queries_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(
        tmp_path, app_source=app_source, selection=selection, judged_query_count=30
    )
    assert "holds 2" in _refuse(path, app_source, selection)


def test_a_report_stating_no_evaluated_count_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(tmp_path, app_source=app_source, selection=selection)
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["judgments"]["evaluated_query_count"]
    path.write_text(json.dumps(document), encoding="utf-8")
    assert "no evaluated query count" in _refuse(path, app_source, selection)


def test_a_report_that_evaluated_no_query_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(
        tmp_path, app_source=app_source, selection=selection, evaluated_query_count=0
    )
    assert "selects 2" in _refuse(path, app_source, selection) or (
        "evaluated no query" in _refuse(path, app_source, selection)
    )


def test_a_report_measured_over_another_selection_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    # A report measured over the first two of three queries, checked against a run
    # that asked for all three: the count and the rows both disagree, and the
    # refusal says which selection this run made.
    _, selection = _judged(tmp_path, queries=3)
    limited = select_queries(selection.path, limit=2)
    assert len(selection.query_ids) == 3 and len(limited.query_ids) == 2
    path = _report(tmp_path, app_source=app_source, selection=limited)
    message = _refuse(path, app_source, selection)
    assert "evaluated 2 queries" in message
    assert "selected 3" in message
    # The same report checked against the selection it was measured over passes, so
    # the refusal is about the disagreement and not about the report.
    facts = _validate(path, app_source, selection=limited)
    assert facts.evaluated_query_count == 2


def test_a_report_naming_another_class_selection_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    # A run that named its classes is checked on the classes the report names, not
    # only on the query set: a report that misnames its own selection would otherwise
    # pass wherever the file happens to make both selections equal.
    _, selection = _judged(tmp_path)
    quoted = select_queries(selection.path, classes="quote")
    path = _report(tmp_path, app_source=app_source, selection=quoted)
    _validate(path, app_source, selection=quoted)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["settings"]["selected_classes"] = ["quote", "entity"]
    path.write_text(json.dumps(document), encoding="utf-8")
    message = _refuse(path, app_source, selection=quoted)
    assert "measured classes quote, entity" in message
    assert "this run asked for quote" in message


def test_a_report_naming_another_skipped_target_set_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    skipped = select_queries(selection.path, skip_targets="t02")
    path = _report(tmp_path, app_source=app_source, selection=skipped)
    _validate(path, app_source, selection=skipped)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["settings"]["skipped_targets"] = []
    path.write_text(json.dumps(document), encoding="utf-8")
    message = _refuse(path, app_source, selection=skipped)
    assert "skipped targets none" in message
    assert "this run skipped t02" in message


def test_a_run_that_named_no_selection_is_not_judged_on_classes(
    tmp_path: Path, app_source: Path
) -> None:
    # A run that named no class measured every class the file holds, so the report
    # is not refused for naming them in the harness's own order.
    _, selection = _judged(tmp_path)
    path = _report(tmp_path, app_source=app_source, selection=selection)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["settings"]["selected_classes"] = ["quote", "entity", "paraphrase"]
    path.write_text(json.dumps(document), encoding="utf-8")
    _validate(path, app_source, selection=selection)


def test_a_mode_missing_one_of_the_selected_queries_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    # A mode with fewer rows is a mean over a different query set than its sibling
    # mode, and the two sit in one table as though they were comparable.
    _, selection = _judged(tmp_path)
    rows = _rows(("bm25",), selection.query_ids, top_k=10, drop=("q02",))
    path = _report(tmp_path, app_source=app_source, selection=selection, rows=rows)
    message = _refuse(path, app_source, selection)
    assert "q02" in message
    assert "missing" in message


def test_a_mode_measuring_a_query_this_run_did_not_select_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    rows = _rows(("bm25",), selection.query_ids + ("q99",), top_k=10)
    path = _report(tmp_path, app_source=app_source, selection=selection, rows=rows)
    message = _refuse(path, app_source, selection)
    assert "q99" in message


def test_a_mode_repeating_a_query_id_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    rows = _rows(("bm25",), selection.query_ids, top_k=10)
    rows.append(dict(rows[0]))
    path = _report(
        tmp_path,
        app_source=app_source,
        selection=selection,
        rows=rows,
        summary={"bm25": {"overall": {"query_count": 3}}},
    )
    message = _refuse(path, app_source, selection)
    assert "same query" in message


def test_a_row_naming_no_query_is_refused(tmp_path: Path, app_source: Path) -> None:
    _, selection = _judged(tmp_path)
    rows = _rows(("bm25",), selection.query_ids, top_k=10)
    del rows[1]["query_id"]
    path = _report(tmp_path, app_source=app_source, selection=selection, rows=rows)
    assert "naming no query" in _refuse(path, app_source, selection)


def test_a_summary_count_that_disagrees_with_its_rows_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    rows = _rows(("bm25",), selection.query_ids, top_k=10)
    path = _report(
        tmp_path,
        app_source=app_source,
        selection=selection,
        rows=rows,
        summary={"bm25": {"overall": {"query_count": 1}}},
    )
    message = _refuse(path, app_source, selection)
    assert "summarises 1 queries" in message


def test_a_summary_stating_no_count_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    path = _report(
        tmp_path,
        app_source=app_source,
        selection=selection,
        summary={"bm25": {"overall": {"mrr": 1.0}}},
    )
    assert "no summary count" in _refuse(path, app_source, selection)


def test_a_row_for_a_mode_nobody_asked_for_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    rows = _rows(("bm25", "dense"), selection.query_ids, top_k=10)
    path = _report(tmp_path, app_source=app_source, selection=selection, rows=rows)
    message = _refuse(path, app_source, selection)
    assert "dense" in message


def test_a_reranked_row_where_one_query_was_not_reranked_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    # One query in two went through the reranker and one did not: the row is a mean
    # over two different kinds of search under one label.
    _, selection = _judged(tmp_path)
    rows = _rows(
        ("hybrid+rerank",),
        selection.query_ids,
        top_k=10,
        reranked=True,
        candidate_count=3,
    )
    rows[1]["reranked"] = False
    path = _report(
        tmp_path,
        app_source=app_source,
        selection=selection,
        modes=("hybrid+rerank",),
        rows=rows,
    )
    message = _refuse(path, app_source, selection, modes=("hybrid+rerank",))
    assert "q02" in message
    assert "other than true" in message


def test_a_reranked_row_that_fell_back_is_refused(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    rows = _rows(
        ("hybrid+rerank",),
        selection.query_ids,
        top_k=10,
        fallback={"reason": "reranker unavailable"},
    )
    path = _report(
        tmp_path,
        app_source=app_source,
        selection=selection,
        modes=("hybrid+rerank",),
        rows=rows,
    )
    message = _refuse(path, app_source, selection, modes=("hybrid+rerank",))
    assert "fell back" in message


def test_a_reranked_row_with_no_candidates_is_not_counted_as_reranked(
    tmp_path: Path, app_source: Path
) -> None:
    # A query that returned nothing never reached a reranker, so its row cannot
    # evidence that one ran. The condition is declared at `candidate_count`; the
    # refusal names it rather than letting the row pass as a reranked search.
    _, selection = _judged(tmp_path)
    rows = _rows(("hybrid+rerank",), selection.query_ids, top_k=10, candidate_count=0)
    path = _report(
        tmp_path,
        app_source=app_source,
        selection=selection,
        modes=("hybrid+rerank",),
        rows=rows,
    )
    message = _refuse(path, app_source, selection, modes=("hybrid+rerank",))
    assert "no candidates" in message
    assert "candidate_count" in message


def test_a_mode_with_no_reranked_row_is_not_judged_for_rerank_evidence(
    tmp_path: Path, app_source: Path
) -> None:
    # A run that asked for bm25 alone cannot be refused for lacking rerank
    # evidence: there was no reranked row in it to be evidence of.
    _, selection = _judged(tmp_path)
    rows = _rows(("bm25",), selection.query_ids, top_k=10, reranked=False)
    path = _report(tmp_path, app_source=app_source, selection=selection, rows=rows)
    facts = _validate(path, app_source, selection=selection)
    assert facts.by_mode["bm25"].rerank_not_applied == selection.query_ids


def test_a_missing_report_is_a_condition_not_a_crash(
    tmp_path: Path, app_source: Path
) -> None:
    _, selection = _judged(tmp_path)
    assert "wrote no report" in _refuse(tmp_path / "absent.json", app_source, selection)


# --- the checkout a code arm measures -----------------------------------------


def _engine_repository(root: Path) -> Path:
    """A minimal git repository that is also a research-rag tree."""

    (root / "scripts").mkdir(parents=True)
    (root / "src" / "research_rag").mkdir(parents=True)
    (root / "src" / "research_rag" / "__init__.py").write_text(
        '"""Fixture engine."""\n', encoding="utf-8"
    )
    (root / "scripts" / "evaluate_retrieval.py").write_text(
        'MODES = ("bm25",)\n', encoding="utf-8"
    )
    (root / "README.md").write_text("fixture engine\n", encoding="utf-8")
    for arguments in (
        ["init", "-q"],
        ["config", "user.email", "tests@example.com"],
        ["config", "user.name", "tests"],
        ["add", "-A"],
        ["commit", "-q", "-m", "first"],
    ):
        subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        )
    return root


def test_a_code_arm_gets_its_own_clone_and_never_writes_to_the_original(
    tmp_path: Path,
) -> None:
    repository = _engine_repository(tmp_path / "engine")
    engine = locate_engine(repository)
    evidence = tmp_path / "run" / "code"
    checkout = make_checkout(
        evidence / "engine",
        app_source=engine,
        base=None,
        patch=None,
        evidence_directory=evidence,
    )
    assert checkout.method == CLONE_METHOD
    assert checkout.root != repository
    assert checkout.base_revision == git_revision_of(repository)
    assert checkout.dirty_before_patch is False
    # The clone is independent: its index and objects are its own, so a checkout,
    # a patch, or a gc in one cannot touch the other.
    assert (checkout.root / ".git").is_dir()
    assert locate_engine(checkout.root).revision == engine.revision
    assert (checkout.root / "README.md").read_text(encoding="utf-8") == (
        "fixture engine\n"
    )


def test_a_code_arms_checkout_captures_the_dirty_state_of_the_tree_it_copied(
    tmp_path: Path,
) -> None:
    repository = _engine_repository(tmp_path / "engine")
    engine = locate_engine(repository)
    # Uncommitted work is what a developer is measuring, so it is captured and
    # applied rather than dropped: the clone is HEAD plus the tree's own changes.
    (repository / "src" / "research_rag" / "patched.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (repository / "README.md").write_text("fixture engine, edited\n", encoding="utf-8")
    (repository / "notes.md").write_text("untracked\n", encoding="utf-8")

    evidence = tmp_path / "run" / "code"
    checkout = make_checkout(
        evidence / "engine",
        app_source=engine,
        base=None,
        patch=None,
        evidence_directory=evidence,
    )
    assert checkout.source_tree_dirty is True
    assert checkout.source_dirty_applied is True
    # `git diff` states tracked changes; an untracked file is stated by its digest
    # in the evidence, and copied into the clone rather than left to chance.
    assert "README.md" in checkout.resulting_diff
    assert (checkout.root / "notes.md").read_text(encoding="utf-8") == "untracked\n"
    # The evidence is kept inside the run, not only inside the checkout.
    captured = json.loads((evidence / SOURCE_EVIDENCE).read_text(encoding="utf-8"))
    assert captured["dirty"] is True
    assert captured["source_revision"] == engine.revision
    # Both untracked files are stated: `patched.py` is untracked too, and a record
    # that named only the one the test happened to think of would understate what
    # the clone carried.
    assert {item["path"] for item in captured["untracked"]} == {
        "notes.md",
        "src/research_rag/patched.py",
    }
    assert all(item["sha256"] for item in captured["untracked"])
    assert (evidence / SOURCE_DIFF).is_file()


def test_an_untracked_symlink_is_refused_rather_than_copied(tmp_path: Path) -> None:
    repository = _engine_repository(tmp_path / "engine")
    (repository / "link.py").symlink_to(repository / "README.md")
    with pytest.raises(ExperimentError) as caught:
        make_checkout(
            tmp_path / "run" / "code" / "engine",
            app_source=locate_engine(repository),
            base=None,
            patch=None,
            evidence_directory=tmp_path / "run" / "code",
        )
    assert "untracked symbolic link" in str(caught.value)


def test_a_code_arms_patch_is_applied_in_the_clone_and_kept_in_the_run(
    tmp_path: Path,
) -> None:
    repository = _engine_repository(tmp_path / "engine")
    engine = locate_engine(repository)
    patch = tmp_path / "variant.patch"
    patch.write_text(
        "diff --git a/src/research_rag/patch.py b/src/research_rag/patch.py\n"
        "new file mode 100644\n"
        "index 0000000..e69de29\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "run" / "code"
    checkout = make_checkout(
        evidence / "engine",
        app_source=engine,
        base=None,
        patch=patch,
        evidence_directory=evidence,
    )
    assert checkout.patch_path == evidence / PATCH_COPY
    assert checkout.patch_path.is_file()
    assert checkout.patch_sha256 == checkout.describe()["patch_sha256"]
    assert (checkout.root / "src" / "research_rag" / "patch.py").exists()
    assert checkout.dirty_before_patch is True


def test_a_tree_that_is_not_a_repository_is_refused_for_a_code_arm(
    app_source: Path, tmp_path: Path
) -> None:
    import shutil

    plain = tmp_path / "not-a-repository"
    shutil.copytree(
        app_source,
        plain,
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"),
        symlinks=True,
    )
    with pytest.raises(ExperimentError) as caught:
        make_checkout(
            tmp_path / "run" / "code" / "engine",
            app_source=locate_engine(plain),
            base=None,
            patch=None,
            evidence_directory=tmp_path / "run" / "code",
        )
    assert "no git repository" in str(caught.value)


def test_a_checkout_that_already_exists_is_refused(tmp_path: Path) -> None:
    repository = _engine_repository(tmp_path / "engine")
    destination = tmp_path / "run" / "code" / "engine"
    destination.mkdir(parents=True)
    with pytest.raises(ExperimentError) as caught:
        make_checkout(
            destination,
            app_source=locate_engine(repository),
            base=None,
            patch=None,
            evidence_directory=destination.parent,
        )
    assert "already exists" in str(caught.value)
