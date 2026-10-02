"""That a specification is refused when it says something unclear.

A specification is the only description of an experiment, so an entry that is
ignored is an experiment nobody ran. Every key is either read or refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_experiments.errors import ExperimentError
from rag_experiments.experiment.spec import (
    ARM_KINDS,
    HARNESS_FIELDS,
    SETTINGS_ARM,
    SPEC_SCHEMA_VERSION,
    load_spec,
)

SETTINGS = {"name": "baseline", "kind": "settings", "overlay": {}}


def _write(path: Path) -> None:
    """Write a judged set shaped like one, so only the split logic is exercised."""

    path.write_text(json.dumps({"schema_version": 1, "queries": []}), encoding="utf-8")


def test_a_complete_specification_reads(make_spec) -> None:
    spec = load_spec(
        make_spec(
            [
                SETTINGS,
                {
                    "name": "deeper",
                    "kind": "settings",
                    "overlay": {"chunking.size": 512},
                },
            ],
            harness={"top_k": 10, "modes": ["bm25", "hybrid"], "offline": True},
        )
    )
    assert SPEC_SCHEMA_VERSION == 1
    assert [arm.name for arm in spec.arms] == ["baseline", "deeper"]
    assert spec.arms[1].overlay == {"chunking.size": 512}
    assert spec.harness["modes"] == ["bm25", "hybrid"]


def test_a_relative_judged_set_resolves_against_the_specification(
    make_spec,
) -> None:
    spec = load_spec(make_spec([SETTINGS]))
    assert len(spec.judgments) == 1
    assert spec.judgments[0].name == "all"
    assert spec.judgments[0].path.is_file()
    assert spec.judgments[0].path.parent == make_spec([SETTINGS]).parent


def test_two_named_splits_are_read_for_one_run(make_spec, tmp_path: Path) -> None:
    # Development and held-out measured together is the point: a held-out number
    # is only comparable with a development one when the same arms produced both.
    held_out = tmp_path / "held-out.json"
    _write(held_out)
    spec = load_spec(
        make_spec(
            [SETTINGS],
            judgments=[
                {"name": "development", "path": "judged.json"},
                {"name": "held-out", "path": held_out.name},
            ],
        )
    )
    assert [item.name for item in spec.judgments] == ["development", "held-out"]
    assert all(item.path.is_file() for item in spec.judgments)


def test_two_splits_of_one_name_are_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(
            make_spec(
                [SETTINGS],
                judgments=[
                    {"name": "queries", "path": "judged.json"},
                    {"name": "queries", "path": "judged.json"},
                ],
            )
        )
    assert "twice" in str(caught.value)


def test_a_split_with_no_name_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([SETTINGS], judgments=[{"path": "judged.json"}]))
    assert "has no name" in str(caught.value)


def test_a_split_with_an_unknown_key_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(
            make_spec(
                [SETTINGS],
                judgments=[{"name": "d", "path": "judged.json", "weight": 1}],
            )
        )
    assert "weight" in str(caught.value)


def test_a_split_that_is_not_an_object_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([SETTINGS], judgments=["judged.json", "held-out.json"]))
    assert "not an object" in str(caught.value)


def test_an_absent_judged_set_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(
            make_spec([SETTINGS], judgments=[{"name": "d", "path": "absent.json"}])
        )
    assert "does not exist" in str(caught.value)


def test_the_specification_states_its_splits_in_its_record_form(
    make_spec,
) -> None:
    described = load_spec(make_spec([SETTINGS])).describe()
    assert described["judgments"] == [
        {"name": "all", "path": described["judgments"][0]["path"]}
    ]


def test_a_specification_that_is_not_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ExperimentError) as caught:
        load_spec(path)
    assert "not JSON" in str(caught.value)


def test_a_specification_that_is_absent_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(tmp_path / "absent.json")
    assert "Cannot read" in str(caught.value)


def test_another_schema_version_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([SETTINGS], schema_version=2))
    assert "schema_version" in str(caught.value)


def test_an_unknown_specification_key_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([SETTINGS], overlays={}))
    assert "overlays" in str(caught.value)
    assert "does not read" in str(caught.value)


def test_an_unknown_arm_key_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([{**SETTINGS, "overlays": {}}]))
    assert "overlays" in str(caught.value)


def test_an_unknown_harness_key_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([SETTINGS], harness={"depth": 50}))
    assert "depth" in str(caught.value)


def test_the_harness_keys_are_built_from_the_ones_the_executor_obeys() -> None:
    from rag_experiments.experiment.spec import (
        HARNESS_JOINED,
        HARNESS_REPEATED,
        HARNESS_SETTINGS,
    )

    assert (
        set(HARNESS_SETTINGS)
        | HARNESS_JOINED
        | HARNESS_REPEATED
        | {
            "top_k",
            "deep_top_k",
            "limit",
        }
        == HARNESS_FIELDS
    )


def test_a_missing_judged_set_is_refused(make_spec) -> None:
    spec_path = make_spec([SETTINGS])
    document = json.loads(spec_path.read_text(encoding="utf-8"))
    document["judgments"] = "absent.json"
    spec_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ExperimentError) as caught:
        load_spec(spec_path)
    assert "judged set does not exist" in str(caught.value)


def test_a_run_with_no_arms_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([]))
    assert "names no arms" in str(caught.value)


def test_two_arms_of_one_name_are_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([SETTINGS, dict(SETTINGS)]))
    assert "twice" in str(caught.value)


def test_an_arm_with_no_name_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([{"kind": "settings", "overlay": {}}]))
    assert "has no name" in str(caught.value)


def test_an_unknown_arm_kind_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([{"name": "a", "kind": "magic"}]))
    assert "magic" in str(caught.value)
    for kind in ARM_KINDS:
        assert kind in str(caught.value)


def test_a_settings_arm_asking_for_a_code_change_is_refused(
    make_spec,
) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([{**SETTINGS, "base": "HEAD"}]))
    assert "settings arm" in str(caught.value)
    assert "code" in str(caught.value)


def test_a_code_arm_that_changes_nothing_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([{"name": "a", "kind": "code"}]))
    assert "changes nothing" in str(caught.value)


def test_a_code_arm_whose_patch_is_absent_is_refused(
    make_spec,
    tmp_path: Path,
) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([{"name": "a", "kind": "code", "patch": "absent.diff"}]))
    assert "has no patch" in str(caught.value)


def test_a_code_arm_whose_patch_is_found_reads_it(
    make_spec,
    tmp_path: Path,
) -> None:
    patch = tmp_path / "variant.diff"
    patch.write_text("--- a\n+++ b\n", encoding="utf-8")
    spec = load_spec(
        make_spec([{"name": "a", "kind": "code", "patch": "variant.diff"}])
    )
    assert spec.arms[0].patch == patch
    assert spec.arms[0].kind == "code"
    assert spec.arms[0].base is None


def test_a_non_object_overlay_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([{**SETTINGS, "overlay": ["chunking.size"]}]))
    assert "non-object overlay" in str(caught.value)


def test_a_non_string_prepare_command_is_refused(make_spec) -> None:
    with pytest.raises(ExperimentError) as caught:
        load_spec(make_spec([{**SETTINGS, "prepare": [{"argv": []}]}]))
    assert "prepare" in str(caught.value)


def test_a_prepare_command_is_read_as_a_list(make_spec) -> None:
    spec = load_spec(
        make_spec(
            [{**SETTINGS, "prepare": ["research-rag ingest --project-root {project}"]}]
        )
    )
    assert spec.arms[0].prepare == ("research-rag ingest --project-root {project}",)


def test_the_joined_harness_keys_are_the_ones_that_take_a_list() -> None:
    assert {"modes", "deep_modes", "classes", "skip_targets"} <= HARNESS_FIELDS
    assert SETTINGS_ARM in ARM_KINDS
