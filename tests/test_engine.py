"""Naming the engine under test, and resolving the settings it declares."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

from rag_experiments import resource_limits
from rag_experiments.engine.harness import harness_contract
from rag_experiments.engine.locate import ENGINE_HARNESS_ENV, locate_engine, tree_digest
from rag_experiments.engine.resolve_settings import (
    SettingsResolutionError,
    _document,
    resolve,
)
from rag_experiments.engine.runner import (
    BYTECODE_ENV,
    CONFIG_HOME_ENV,
    SETTINGS_PREFIX,
    SETTINGS_PREFIXES,
    child_environment,
    describe_child_environment,
)
from rag_experiments.errors import ExperimentError


def test_a_directory_without_the_package_is_not_an_engine(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError) as caught:
        locate_engine(tmp_path)
    assert "research-rag tree" in str(caught.value)
    assert "scripts/evaluate_retrieval.py" in str(caught.value)


def test_a_missing_path_is_a_condition_not_a_crash(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError) as caught:
        locate_engine(tmp_path / "absent")
    assert "not a directory" in str(caught.value)


def test_naming_an_engine_states_its_revision_harness_and_content(
    app_source: Path,
) -> None:
    engine = locate_engine(app_source)
    assert engine.root == app_source
    assert engine.harness.is_file()
    assert engine.harness.name == "evaluate_retrieval.py"
    described = engine.describe()
    # A tree that is not a checkout is reported with no revision rather than
    # refused, so the record can say that is why a number has no commit.
    assert "revision" in described
    assert "dirty" in described
    assert described["harness"] == str(engine.harness)
    if engine.revision is not None:
        assert len(engine.revision) == 40
    assert (app_source / "src").is_dir()
    assert engine.source_relative == app_source / "src"


def test_the_content_digest_covers_what_a_measurement_is_moved_by(
    app_source: Path, tmp_path: Path
) -> None:
    before = tree_digest(app_source)
    assert before and len(before) == 64
    assert tree_digest(app_source) == before, "the digest must be stable"

    # A tree holding none of the digested paths has no such value rather than a
    # digest of nothing, which would compare equal to any other empty tree.
    empty = tmp_path / "not-an-engine"
    empty.mkdir()
    assert tree_digest(empty) is None

    # A change to a file outside the digested paths moves nothing, and a change to
    # one inside it moves the digest.
    import shutil

    clone = tmp_path / "engine"
    shutil.copytree(
        app_source,
        clone,
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"),
        symlinks=True,
    )
    clone_digest = tree_digest(clone)
    assert clone_digest == before
    (clone / "notes.md").write_text("not code\n", encoding="utf-8")
    assert tree_digest(clone) == before
    package = next((clone / "src" / "research_rag").glob("*.py"))
    package.write_text(
        package.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8"
    )
    assert tree_digest(clone) != before


def test_the_harness_contract_is_read_from_the_engine_not_declared_here(
    app_source: Path,
) -> None:
    contract = harness_contract(locate_engine(app_source))
    described = contract.describe()
    assert contract.harness == app_source / "scripts" / "evaluate_retrieval.py"
    assert "hybrid+rerank" in contract.modes
    assert contract.rerank_mode == "hybrid+rerank"
    assert contract.report_schema_version >= 1
    # A reranker named on the command line is a second row for one mode, and both
    # rows answer for that mode.
    assert contract.rerank_modes(contract.modes) == ("hybrid+rerank",)
    assert contract.rerank_modes(
        ("bm25", "hybrid+rerank[fastembed/rerank-embed-english-v3.0]")
    ) == ("hybrid+rerank[fastembed/rerank-embed-english-v3.0]",)
    assert contract.matches("hybrid+rerank", "hybrid+rerank[bge-reranker-base]")
    assert not contract.matches("hybrid", "hybrid+rerank")
    assert described["report_schema_version"] == contract.report_schema_version


def test_a_harness_that_declares_nothing_to_measure_is_refused(
    app_source: Path, tmp_path: Path
) -> None:
    tree = tmp_path / "engine"
    (tree / "scripts").mkdir(parents=True)
    (tree / "src" / "research_rag").mkdir(parents=True)
    (tree / "src" / "research_rag" / "__init__.py").write_text("", encoding="utf-8")
    (tree / "scripts" / "evaluate_retrieval.py").write_text(
        "print('no modes here')\n", encoding="utf-8"
    )
    with pytest.raises(ExperimentError) as caught:
        harness_contract(locate_engine(tree))
    assert "MODES" in str(caught.value)


def test_the_pinned_document_is_readable_by_the_apps_own_reader(
    project: Path, app_source: Path, tmp_path: Path
) -> None:
    answer = resolve(project, {"retrieval.rrf_k": 120})
    document = tomllib.loads(answer["document"])
    assert document["retrieval"]["rrf_k"] == 120

    # The app is the reader that matters, so the app's own layer machinery reads
    # the file back and says which layer supplied each value.
    from research_rag.project.settings import SETTINGS, sources_for
    from research_rag.project.settings_layers import resolve_settings

    sources = sources_for(project)
    pinned = tmp_path / "config.toml"
    pinned.write_text(answer["document"], encoding="utf-8")
    _values, provenance = resolve_settings(
        SETTINGS,
        type(sources)(
            default_file=sources.default_file,
            user_config=None,
            project_root=project,
            project_config=pinned,
        ),
        environ={},
    )
    assert provenance["retrieval.rrf_k"].startswith("project config")
    assert provenance["chunking.size"].startswith("project config")


def test_every_declared_setting_is_pinned_rather_than_only_the_overlay(
    project: Path, app_source: Path
) -> None:
    from research_rag.project.settings import SETTINGS

    answer = resolve(project, {})
    assert set(answer["values"]) == {setting.key for setting in SETTINGS}
    assert answer["overridden"] == []


def test_a_baseline_layer_is_reported_for_every_key(
    project: Path, app_source: Path
) -> None:
    answer = resolve(project, {})
    assert set(answer["baseline_layers"]) == set(answer["values"])
    assert all(layer for layer in answer["baseline_layers"].values())


def test_a_baseline_layer_is_where_the_key_came_from_before_the_arm_overrode_it(
    project: Path,
) -> None:
    # The overlay is the command-line layer, so a resolution that carried the
    # overlay reports the command line for every overridden key. That is true of
    # the arm and says nothing about what it was compared against, which is why
    # the baseline is resolved separately and reported beside it.
    answer = resolve(project, {"retrieval.rrf_k": 120})
    assert answer["overridden"] == ["retrieval.rrf_k"]
    assert answer["arm_layers"]["retrieval.rrf_k"].startswith("command line")
    assert not answer["baseline_layers"]["retrieval.rrf_k"].startswith("command")
    assert answer["baseline_values"]["retrieval.rrf_k"] != 120
    assert answer["values"]["retrieval.rrf_k"] == 120
    # A key the arm did not touch has the same layer in both resolutions.
    assert (
        answer["arm_layers"]["chunking.size"]
        == answer["baseline_layers"]["chunking.size"]
    )


def test_the_account_overlay_is_part_of_the_baseline(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A baseline is what a reader gets today, and that includes the account's own
    # settings. Resolving it with the configuration home relocated would report the
    # packaged defaults for an account that has configured the app.
    home = tmp_path / "account"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    import platformdirs

    overlay = platformdirs.user_config_path("research-rag") / "config.toml"
    assert overlay.is_relative_to(home)
    overlay.parent.mkdir(parents=True)
    overlay.write_text("[retrieval]\nrrf_k = 42\n", encoding="utf-8")

    answer = resolve(project, {})
    assert answer["values"]["retrieval.rrf_k"] == 42
    assert answer["baseline_layers"]["retrieval.rrf_k"].startswith("user config")


def test_an_environment_variable_cannot_reach_the_pinned_document(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The environment layer is dropped for both resolutions, so a value exported in
    # the reader's shell cannot outrank a project file that states every value.
    monkeypatch.setenv("RESEARCH_RAG_RETRIEVAL_RRF_K", "999")
    answer = resolve(project, {})
    assert answer["values"]["retrieval.rrf_k"] != 999
    assert not answer["baseline_layers"]["retrieval.rrf_k"].startswith("environment")


def test_the_model_cache_is_pinned_as_an_absolute_path(
    project: Path, app_source: Path, tmp_path: Path
) -> None:
    # A sandbox is given an empty configuration home, so an empty or relative cache
    # setting would leave the copy looking for binaries in a directory it made and
    # downloading what a run must not download. `tmp_path` stands in for the
    # sandbox: the cache is resolved against the tree the values will be pinned into.
    answer = resolve(project, {}, pin_project=tmp_path)
    cache = answer["model_cache_root"]
    assert cache
    assert Path(cache).is_absolute()
    assert answer["values"]["runtime.model_cache_root"] == cache
    assert tomllib.loads(answer["document"])["runtime"]["model_cache_root"] == cache
    assert Path(cache).is_dir(), "the shared cache must be the one that exists"


def test_an_armed_model_cache_path_is_resolved_rather_than_copied_verbatim(
    project: Path, tmp_path: Path
) -> None:
    answer = resolve(project, {"runtime.model_cache_root": "~/models"})
    pinned = answer["values"]["runtime.model_cache_root"]
    assert pinned == str(Path("~/models").expanduser().resolve())


def test_an_unknown_setting_is_refused_by_the_engine_not_here(
    project: Path, app_source: Path
) -> None:
    with pytest.raises(SettingsResolutionError) as caught:
        resolve(project, {"retrieval.no_such_knob": 1})
    assert "retrieval.no_such_knob" in str(caught.value)


def test_an_out_of_range_value_is_refused(project: Path, app_source: Path) -> None:
    with pytest.raises(SettingsResolutionError):
        resolve(project, {"retrieval.rrf_k": 10_000})


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, "true"), (False, "false"), ("en", '"en"'), (60, "60")],
)
def test_every_toml_value_kind_round_trips(value: object, expected: str) -> None:
    document = tomllib.loads(_document({"retrieval.sample": value}))
    assert document["retrieval"]["sample"] == value
    assert expected


def test_a_value_toml_cannot_carry_is_refused() -> None:
    with pytest.raises(SettingsResolutionError) as caught:
        _document({"retrieval.sample": ["a", "list"]})
    assert "TOML cannot carry" in str(caught.value)


def test_the_child_selects_the_tree_and_drops_the_settings_layer(
    app_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESEARCH_RAG_CHUNKING_SIZE", "999")
    monkeypatch.setenv("RESEARCH_ULTRARAG_RUNTIME_ROOT", "/somewhere/else")
    monkeypatch.setenv("UNRELATED", "kept")
    engine = locate_engine(app_source)
    environment = child_environment(engine, config_home=tmp_path)
    assert environment["UNRELATED"] == "kept"
    assert not any(name.startswith(SETTINGS_PREFIX) for name in environment), (
        "an exported RESEARCH_RAG_ variable would outrank the pinned file"
    )
    # The harness reads its cache root, runtime root, and dense backend from
    # RESEARCH_ULTRARAG_ variables, so an exported one would relocate the state a
    # measurement reads without appearing anywhere in the record.
    assert "RESEARCH_ULTRARAG_RUNTIME_ROOT" not in environment
    assert not any(
        name.startswith(prefix) for name in environment for prefix in SETTINGS_PREFIXES
    )
    assert str(engine.source_relative) in environment[ENGINE_HARNESS_ENV]
    assert environment[CONFIG_HOME_ENV] == str(tmp_path)
    # The model cache is deliberately shared: those are immutable binaries.
    assert "XDG_CACHE_HOME" not in environment


def test_a_child_writes_no_bytecode_beside_the_tree_it_reads(
    app_source: Path,
) -> None:
    # The engine under test is frequently a developer's working checkout, and an
    # import must not leave a __pycache__ in the code the run measured.
    environment = child_environment(locate_engine(app_source))
    assert environment[BYTECODE_ENV] == "1"


def test_the_record_states_what_a_child_environment_changes(
    app_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RESEARCH_RAG_RRF_K", "1")
    monkeypatch.setenv("RESEARCH_ULTRARAG_DENSE_BACKEND", "exact")
    for name in resource_limits.THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    described = describe_child_environment(
        locate_engine(app_source), config_home=tmp_path
    )
    assert described["dropped_variables"] == [
        "RESEARCH_RAG_RRF_K",
        "RESEARCH_ULTRARAG_DENSE_BACKEND",
    ]
    assert described["given"][BYTECODE_ENV] == "1"
    assert described["given"][CONFIG_HOME_ENV] == str(tmp_path)
    # The numerical thread pools are capped by default, and those names are
    # reported as set; a reader's own value for one of them is left alone.
    assert set(described["given"]) == {
        ENGINE_HARNESS_ENV,
        BYTECODE_ENV,
        CONFIG_HOME_ENV,
        *resource_limits.THREAD_ENV_VARS,
    }
    assert described["resource_limits"]["capability"]["available"] is True


def test_a_tree_precedes_the_interpreters_own_path(app_source: Path) -> None:
    engine = locate_engine(app_source)
    environment = child_environment(engine)
    first = environment[ENGINE_HARNESS_ENV].split(":")[0]
    assert first == str(engine.source_relative)


def test_the_git_questions_asked_of_a_working_tree_do_not_write_to_it(
    app_source: Path,
) -> None:
    # `git status` may refresh the index it reads. With optional locks allowed that
    # is a write into the developer's checkout, which is the tree under test.
    engine = locate_engine(app_source)
    index = app_source / ".git" / "index"
    before = index.stat().st_mtime_ns if index.is_file() else None
    completed = subprocess.run(
        ["git", "-C", str(app_source), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "GIT_OPTIONAL_LOCKS": "0"},
    )
    assert completed.returncode == 0
    after = index.stat().st_mtime_ns if index.is_file() else None
    assert before == after
    assert engine.revision is not None
