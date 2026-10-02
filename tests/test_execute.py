"""That an arm's command says what the specification asked for.

The measurement is the app's own harness, run as a subprocess. These tests hold
the part this harness owns: which flags reach that harness, which settings reach
the pinned file, and that a shape the app's parser does not have is refused
rather than guessed at.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FIRST_GENERATION

from rag_experiments.engine.locate import locate_engine
from rag_experiments.errors import ExperimentError
from rag_experiments.experiment.execute import (
    HARNESS_JOINED,
    HARNESS_REPEATED,
    HARNESS_SETTINGS,
    _harness_flag,
    _interpolate,
    _segment,
    effective_overlay,
    prepare_arm,
)
from rag_experiments.experiment.spec import Arm, JudgedSet, RunSpec


def _spec(**overrides: object) -> RunSpec:
    base: dict[str, object] = {
        "path": Path("/tmp/spec.json"),
        "name": "fixture",
        "source_project": Path("/tmp/corpus"),
        "judgments": Path("/tmp/judged.json"),
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
        str(tmp_path / "sources"),
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
    project: Path, workspace: Path, app_source: Path
) -> None:
    spec = _spec(
        name="fixture",
        source_project=project,
        judgments=Path("/tmp/judged.json"),
        harness={"top_k": 10, "offline": True},
        arms=(_arm(overlay={"retrieval.rrf_k": 120}),),
    )
    prepared = prepare_arm(
        spec,
        spec.arms[0],
        workspace=workspace,
        run_directory=Path("/tmp"),
        app_source=locate_engine(app_source),
    )
    document = (prepared.sandbox.settings_file).read_text(encoding="utf-8")
    assert "rrf_k = 120" in document
    assert "offline = true" in document, "the harness block is a setting too"
    assert prepared.settings["key_count"] > 30
    assert sorted(prepared.settings["overridden"]) == [
        "retrieval.rrf_k",
        "runtime.offline",
    ]
    assert prepared.sandbox.generation_ids == (FIRST_GENERATION,)
    assert prepared.settings["document_sha256"]
    assert prepared.engine.root == app_source


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
    prepared = prepare_arm(
        spec,
        spec.arms[0],
        workspace=workspace,
        run_directory=Path("/tmp"),
        app_source=locate_engine(app_source),
    )
    # A settings arm runs the engine as it stands, with no checkout of its own.
    assert prepared.checkout is None
    assert prepared.engine.root == app_source
    assert prepared.settings["overridden"] == []


class _StubSandbox:
    """The three attributes `_interpolate` reads, so it can be tested alone."""

    def __init__(self, root: Path) -> None:
        self.root = root / "project"
        self.source_root = root / "sources"
        self.workspace = root / "sandbox"


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


def test_a_run_record_is_json_serialisable_whole(
    project: Path, workspace: Path, app_source: Path
) -> None:
    spec = _spec(name="fixture", source_project=project, arms=(_arm(),))
    prepared = prepare_arm(
        spec,
        spec.arms[0],
        workspace=workspace,
        run_directory=Path("/tmp"),
        app_source=locate_engine(app_source),
    )
    # A record the reader cannot load is a record nobody can act on.
    assert json.loads(json.dumps(prepared.describe()))["sandbox"]["name"] == (
        "fixture-baseline"
    )


def _two_split_spec(project: Path, tmp_path: Path) -> RunSpec:
    for name in ("development.json", "held-out.json"):
        (tmp_path / name).write_text(
            json.dumps({"schema_version": 1, "queries": []}), encoding="utf-8"
        )
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


def test_the_validation_command_carries_the_apps_own_flag(
    app_source: Path, project: Path, tmp_path: Path
) -> None:
    from rag_experiments.experiment.execute import VALIDATE_FLAG, _measure_command

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
    # A split name is a record key and a printed label, so it is not trusted to be
    # a safe file name.
    assert REPORT_PATTERN.format(split=_segment("held out/query")) == (
        "report-held-out-query.json"
    )


def test_a_split_name_that_is_not_a_path_stays_one_segment() -> None:
    from rag_experiments.experiment.execute import _segment

    assert _segment("../../etc/passwd") == "etc-passwd"
    assert _segment("///") == "split"


def test_a_measured_arm_reports_one_entry_per_split(
    project: Path, workspace: Path, app_source: Path, tmp_path: Path
) -> None:
    prepared = prepare_arm(
        _two_split_spec(project, tmp_path),
        _arm(),
        workspace=workspace,
        run_directory=tmp_path,
        app_source=locate_engine(app_source),
    )
    # Preparation is split-agnostic: one sandbox, one pinned file, one engine, and
    # the splits differ only in which judged set each measurement is handed.
    assert prepared.sandbox.generation_ids == (FIRST_GENERATION,)
    assert prepared.settings["key_count"] > 30
