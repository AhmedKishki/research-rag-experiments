"""Partition inputs by family without inventing exclusions or held-out status."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "examples" / "make-splits.py"


@pytest.fixture
def partitioner():
    spec = importlib.util.spec_from_file_location("make_splits", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample():
    return {
        "schema_version": 1,
        "targets": [{"target_id": key} for key in ("t1", "t2", "t3", "t12")],
        "queries": [
            {"query_id": "q1", "target_id": "t1", "class": "quote"},
            {"query_id": "q2", "target_id": "t1", "class": "paraphrase"},
            {"query_id": "q3", "target_id": "t2", "class": "quote"},
            {"query_id": "q4", "target_id": "t3", "class": "entity"},
            {"query_id": "q5", "target_id": "t12", "class": "quote"},
        ],
    }


def group_map(*outputs):
    return {
        target: group
        for output in outputs
        for group, members in output["document"]["partition_metadata"][
            "target_groups"
        ].items()
        for target in members
    }


def test_targets_match_only_the_queries_of_each_partition(partitioner):
    original = sample()
    before = json.dumps(original, sort_keys=True)
    first, second, exclusions = partitioner.split(original)
    sets = []
    for output in (first, second):
        document = output["document"]
        declared = {t["target_id"] for t in document["targets"]}
        queried = {q["target_id"] for q in document["queries"]}
        assert declared == queried
        assert document["partition_metadata"]["role"] == "exploratory"
        sets.append(declared)
    assert not sets[0] & sets[1]
    assert sets[0] | sets[1] == {"t1", "t2", "t3", "t12"}
    assert not exclusions  # No hard-coded benchmark-specific exclusion.
    assert json.dumps(original, sort_keys=True) == before


def test_explicit_exclusion_removes_targets_and_queries(partitioner):
    first, second, exclusions = partitioner.split(sample(), {"t12": "source absent"})
    assert exclusions == {"t12": "source absent"}
    for output in (first, second):
        document = output["document"]
        assert all(t["target_id"] != "t12" for t in document["targets"])
        assert all(q["target_id"] != "t12" for q in document["queries"])
        assert document["partition_metadata"]["excluded_targets"] == exclusions


def test_question_family_spanning_targets_stays_together(partitioner):
    document = sample()
    document["queries"][0]["family_id"] = "question-a"
    document["queries"][2]["family_id"] = "question-a"
    first, second, _ = partitioner.split(document)
    memberships = {
        q["target_id"]: output["name"]
        for output in (first, second)
        for q in output["document"]["queries"]
    }
    assert memberships["t1"] == memberships["t2"]
    groups = group_map(first, second)
    assert groups["t1"] == groups["t2"]


def test_exclusion_requires_reason_and_existing_target(partitioner):
    with pytest.raises(ValueError, match="nonempty"):
        partitioner.split(sample(), {"t12": ""})
    with pytest.raises(ValueError, match="Unknown"):
        partitioner.split(sample(), {"missing": "source absent"})
    with pytest.raises(ValueError, match="ID=reason"):
        partitioner.exclusion_map(["t12"])


def test_output_manifest_records_input_and_exclusions(partitioner, tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(sample()), encoding="utf-8")
    out = tmp_path / "partitions"
    assert (
        partitioner.main(
            [
                "--judged",
                str(source),
                "--out",
                str(out),
                "--exclude-target",
                "t12=source absent",
            ]
        )
        == 0
    )
    manifest = json.loads((out / "partition-manifest.json").read_text())
    assert manifest["role"] == "exploratory"
    assert manifest["excluded_targets"] == {"t12": "source absent"}
    assert len(manifest["input_sha256"]) == 64
    assert sum(p["query_count"] for p in manifest["partitions"]) == 4
    with pytest.raises(SystemExit):
        partitioner.main(["--judged", str(source), "--out", str(out)])


def test_both_target_and_query_family_links_are_preserved(partitioner):
    document = sample()
    document["targets"][0]["family_id"] = "shared"
    document["targets"][1]["family_id"] = "shared"
    document["queries"][0]["family_id"] = "sub-one"
    document["queries"][1]["family_id"] = "sub-one"
    document["queries"][2]["family_id"] = "sub-two"
    first, second, _ = partitioner.split(document)
    location = {
        t["target_id"]: output["name"]
        for output in (first, second)
        for t in output["document"]["targets"]
    }
    assert location["t1"] == location["t2"]
    groups = group_map(first, second)
    assert groups["t1"] == groups["t2"]


def test_family_label_cannot_implicitly_match_an_unrelated_target_id(partitioner):
    document = sample()
    document["queries"][0]["family_id"] = "t2"
    first, second, _ = partitioner.split(document)
    groups = group_map(first, second)
    assert groups["t1"] != groups["t2"]


def test_unqueried_targets_are_accounted_for_without_being_scored(partitioner):
    document = sample()
    document["targets"].append({"target_id": "orphan"})
    first, second, excluded = partitioner.split(document)
    assert not excluded
    for output in (first, second):
        assert output["document"]["partition_metadata"]["unqueried_targets"] == [
            "orphan"
        ]
        assert all(t["target_id"] != "orphan" for t in output["document"]["targets"])


def test_partitions_do_not_share_mutable_input_objects(partitioner):
    document = sample()
    document["extra"] = {"value": []}
    first, second, _ = partitioner.split(document)
    first["document"]["extra"]["value"].append("changed")
    first["document"]["queries"][0]["class"] = "changed"
    assert document["extra"]["value"] == []
    assert second["document"]["extra"]["value"] == []
    assert all(q["class"] != "changed" for q in document["queries"])
