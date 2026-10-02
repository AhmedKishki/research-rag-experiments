"""Naming the engine under test, and resolving the settings it declares."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from rag_experiments.engine.locate import ENGINE_HARNESS_ENV, locate_engine
from rag_experiments.engine.resolve_settings import (
    SettingsResolutionError,
    _document,
    resolve,
)
from rag_experiments.engine.runner import (
    CONFIG_HOME_ENV,
    SETTINGS_PREFIX,
    child_environment,
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


def test_naming_an_engine_states_its_revision_and_harness(app_source: Path) -> None:
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


def test_the_pinned_document_is_readable_by_the_apps_own_reader(
    project: Path, app_source: Path, tmp_path: Path
) -> None:
    answer = resolve(app_source, {"retrieval.rrf_k": 120})
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

    answer = resolve(app_source, {})
    assert set(answer["values"]) == {setting.key for setting in SETTINGS}
    assert answer["overridden"] == []


def test_a_baseline_layer_is_reported_for_every_key(
    project: Path, app_source: Path
) -> None:
    answer = resolve(app_source, {})
    assert set(answer["baseline_layers"]) == set(answer["values"])
    assert all(layer for layer in answer["baseline_layers"].values())


def test_an_unknown_setting_is_refused_by_the_engine_not_here(
    project: Path, app_source: Path
) -> None:
    with pytest.raises(SettingsResolutionError) as caught:
        resolve(app_source, {"retrieval.no_such_knob": 1})
    assert "retrieval.no_such_knob" in str(caught.value)


def test_an_out_of_range_value_is_refused(project: Path, app_source: Path) -> None:
    with pytest.raises(SettingsResolutionError):
        resolve(app_source, {"retrieval.rrf_k": 10_000})


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
    monkeypatch.setenv("UNRELATED", "kept")
    engine = locate_engine(app_source)
    environment = child_environment(engine, config_home=tmp_path)
    assert environment["UNRELATED"] == "kept"
    assert not any(name.startswith(SETTINGS_PREFIX) for name in environment), (
        "an exported RESEARCH_RAG_ variable would outrank the pinned file"
    )
    assert str(engine.source_relative) in environment[ENGINE_HARNESS_ENV]
    assert environment[CONFIG_HOME_ENV] == str(tmp_path)
    # The model cache is deliberately shared: those are immutable binaries.
    assert "XDG_CACHE_HOME" not in environment


def test_a_tree_precedes_the_interpreters_own_path(app_source: Path) -> None:
    engine = locate_engine(app_source)
    environment = child_environment(engine)
    first = environment[ENGINE_HARNESS_ENV].split(":")[0]
    assert first == str(engine.source_relative)
