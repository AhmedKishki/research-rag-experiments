"""Pool unions preserve provenance, hide retrieval hints, and refuse changed inputs."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from rag_experiments.annotation import build_pool, check_annotations, load_package
from rag_experiments.errors import ExperimentError


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def archive(tmp_path):
    project = tmp_path / "retained-project"
    portable = project / ".research-rag"
    generation = "20260101T000000Z-aabbccdd"
    root = portable / "runtime" / "generations" / generation
    write(
        portable / "project.json",
        {
            "schema_version": 1,
            "project_id": "121304a5-9c33-4df0-b768-712df11b772a",
            "name": "fixture",
            "source_directory": "sources",
        },
    )
    files, documents, catalog = [], [], {}
    for source_id in ("source-one", "source-two"):
        relative = source_id + ".pdf"
        original = project / "sources" / relative
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(b"%PDF-1.7\nfixture original " + source_id.encode())
        files.append(
            {
                "source_id": source_id,
                "source_relative_path": relative,
                "source_path": "sources/" + relative,
                "sha256": sha(original),
                "included": True,
            }
        )
        documents.append(
            {
                "document_id": "doc-" + source_id,
                "source_id": source_id,
                "title": "Source " + source_id,
                "authors": [source_id],
                "year": "2000",
            }
        )
        catalog[source_id] = {"source_relative_path": relative}
    write(portable / "source-catalog.json", {"schema_version": 1, "sources": catalog})
    write(portable / "source-metadata.json", {"schema_version": 1, "sources": {}})
    write(portable / "source-exclusions.json", {"schema_version": 1, "sources": {}})
    chunks = []
    for cid, source_id in (
        ("INTERNAL_GOLD", "source-one"),
        ("INTERNAL_COPY", "source-two"),
        ("INTERNAL_CONTEXT", "source-one"),
    ):
        chunks.append(
            {
                "chunk_id": cid,
                "document_id": "doc-" + source_id,
                "source_id": source_id,
                "contents": "The shared passage"
                if cid != "INTERNAL_CONTEXT"
                else "Context material",
                "locator": {"type": "pdf_page", "page": 1},
            }
        )
    chunk_file = root / "chunks" / "chunks.jsonl"
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    chunk_file.write_text(
        "".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8"
    )
    lookup = root / "indexes" / "artifact-lookup.sqlite3"
    lookup.parent.mkdir(parents=True)
    with sqlite3.connect(lookup) as db:
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        db.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [("chunks_sha256", sha(chunk_file)), ("chunks_count", "3")],
        )
    write(
        root / "manifest.json",
        {
            "generation_id": generation,
            "files": {"chunks": "chunks/chunks.jsonl"},
            "documents": documents,
            "source_files": files,
        },
    )
    write(portable / "runtime" / "current.json", {"generation_id": generation})
    run = tmp_path / "retained-run"
    judged = run / "judged" / "exploratory.json"
    write(
        judged,
        {
            "schema_version": 1,
            "targets": [{"target_id": "SECRET_TARGET"}],
            "queries": [
                {
                    "query_id": "SECRET_QUERY",
                    "target_id": "SECRET_TARGET",
                    "query": "What is the passage about?",
                    "class": "paraphrase",
                }
            ],
            "partition_metadata": {"excluded_targets": {"removed": "source absent"}},
        },
    )
    report = run / "SECRET_ARM" / "report.json"
    write(
        report,
        {
            "schema_version": 3,
            "project": {"generation_id": generation},
            "harness": {"script_sha256": "a" * 64},
            "targets": [
                {
                    "target_id": "SECRET_TARGET",
                    "chunk_id": "INTERNAL_GOLD",
                    "chunk_id_at_measurement": "STALE_TARGET_ID",
                }
            ],
            "runs": [
                {
                    "query_id": "SECRET_QUERY",
                    "query": "What is the passage about?",
                    "target_id": "SECRET_TARGET",
                    "family_id": "SECRET_TARGET",
                    "mode": "hybrid+rerank",
                    "top_k": 2,
                    "candidate_depth": 40,
                    "rerank_window": 2,
                    "rerank_requested": True,
                    "reranked": True,
                    "rerank_fallback": None,
                    "target_chunk_id": "INTERNAL_GOLD",
                    "returned_chunk_ids": ["INTERNAL_COPY", "INTERNAL_CONTEXT"],
                    "collapsed_pairs": [
                        {
                            "chunk_id": "INTERNAL_GOLD",
                            "repeated_chunk_id": "INTERNAL_COPY",
                            "collapsed_by": "same_words",
                            "similarity": 0.999,
                        }
                    ],
                    "collapsed_pairs_truncated": False,
                }
            ],
        },
    )
    record = {
        "schema_version": 2,
        "verdict": "verified",
        "run": {"kind": "measurement", "complete": True},
        "source_project": {
            "project_root": str(tmp_path / "live-original"),
            "guard": {"state": "unchanged"},
        },
        "engine": {"before": {"root": str(tmp_path / "engine")}, "changed": False},
        "judgments": [
            {"name": "exploratory", "kept_at": str(judged), "kept_sha256": sha(judged)}
        ],
        "arms": [
            {
                "arm": {"name": "SECRET_ARM"},
                "sandbox": {"root": str(project)},
                "reports": {"exploratory": str(report)},
                "measure": [
                    {
                        "split": "exploratory",
                        "exit_code": 0,
                        "report": str(report),
                        "report_sha256": sha(report),
                    }
                ],
            }
        ],
    }
    write(run / "run.json", record)
    spec = tmp_path / "pool-spec.json"
    write(
        spec,
        {
            "schema_version": 1,
            "name": "pilot",
            "seed": 42,
            "inputs": [
                {
                    "run": str(run),
                    "arms": ["SECRET_ARM"],
                    "splits": ["exploratory"],
                    "modes": ["hybrid+rerank"],
                }
            ],
            "repeat_fraction": 0.25,
            "max_pairs_per_query": 3,
        },
    )
    return {
        "spec": spec,
        "report": report,
        "run": run,
        "project": project,
        "chunks": chunk_file,
        "output": tmp_path / "private-pool",
        "judged": judged,
    }


def test_union_includes_missed_target_and_keeps_other_authors(archive):
    summary = build_pool(archive["spec"], archive["output"])
    loaded = load_package(archive["output"])
    assert summary["status"] == "pending_human_judgments"
    assert summary["policy_ready"] is False
    assert len(loaded["pool"]["items"]) == 3
    same = [i for i in loaded["pool"]["items"] if i["passage"] == "The shared passage"]
    assert len(same) == 2
    assert (
        same[0]["source"]["source_relative_path"]
        != same[1]["source"]["source_relative_path"]
    )
    private = json.loads((archive["output"] / "private" / "key.json").read_text())
    assert {i["chunk_id"] for i in private["item_map"].values()} == {
        "INTERNAL_GOLD",
        "INTERNAL_COPY",
        "INTERNAL_CONTEXT",
    }


def test_reviewer_data_hides_all_provenance_hints(archive):
    build_pool(archive["spec"], archive["output"])
    reviewer = archive["output"] / "reviewer"
    public = "\n".join(p.read_text() for p in reviewer.glob("*") if p.is_file())
    for secret in (
        "SECRET_ARM",
        "SECRET_TARGET",
        "SECRET_QUERY",
        "INTERNAL_GOLD",
        "INTERNAL_COPY",
        "same_words",
        str(archive["run"]),
        str(archive["project"]),
    ):
        assert secret not in public
    result = check_annotations(archive["output"], reviewer / "template.json")
    assert result["status"] == "incomplete"
    template = json.loads((reviewer / "template.json").read_text())
    assert all(
        i["relevance"] is None
        and i["usability"] is None
        and i["source_verified"] is None
        for i in template["items"]
    )


def test_originals_are_independent_and_inputs_are_unchanged(archive):
    inputs = list(archive["project"].rglob("*")) + list(archive["run"].rglob("*"))
    before = {p: sha(p) for p in inputs if p.is_file()}
    build_pool(archive["spec"], archive["output"])
    assert before == {p: sha(p) for p in before}
    for original in (archive["output"] / "reviewer" / "originals").iterdir():
        assert original.stat().st_ino not in {
            p.stat().st_ino for p in (archive["project"] / "sources").iterdir()
        }
    assert archive["output"].stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("field", ["report", "judged", "chunks"])
def test_tampered_inputs_are_refused_without_partial_package(archive, field):
    path = archive[field]
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ExperimentError, match="integrity check failed"):
        build_pool(archive["spec"], archive["output"])
    assert not archive["output"].exists()


def test_changed_original_is_refused(archive):
    original = next((archive["project"] / "sources").glob("*.pdf"))
    original.write_bytes(b"changed")
    with pytest.raises(ExperimentError, match="integrity check failed"):
        build_pool(archive["spec"], archive["output"])


def test_output_cannot_overlap_retained_or_live_inputs(archive):
    for output in (
        archive["run"] / "nested",
        archive["project"] / "nested",
        archive["output"].parent / "live-original" / "nested",
    ):
        with pytest.raises(ExperimentError, match="annotation output"):
            build_pool(archive["spec"], output)


def test_existing_package_is_not_overwritten(archive):
    build_pool(archive["spec"], archive["output"])
    manifest = archive["output"] / "private" / "manifest.json"
    before = sha(manifest)
    with pytest.raises(ExperimentError, match="already exists"):
        build_pool(archive["spec"], archive["output"])
    assert sha(manifest) == before


def test_mandatory_union_is_not_silently_truncated(archive):
    spec = json.loads(archive["spec"].read_text())
    spec["max_candidates_per_query"] = 2
    write(archive["spec"], spec)
    with pytest.raises(ExperimentError, match="Mandatory union"):
        build_pool(archive["spec"], archive["output"])
    assert not archive["output"].exists()


def test_missing_canonical_candidate_cannot_become_empty_text(archive):
    report = json.loads(archive["report"].read_text())
    report["runs"][0]["returned_chunk_ids"].append("NONEXISTENT_CHUNK")
    write(archive["report"], report)
    record = json.loads((archive["run"] / "run.json").read_text())
    record["arms"][0]["measure"][0]["report_sha256"] = sha(archive["report"])
    write(archive["run"] / "run.json", record)
    with pytest.raises(ExperimentError, match="Canonical text coverage"):
        build_pool(archive["spec"], archive["output"])


def test_modified_bundle_fails_validation(archive):
    build_pool(archive["spec"], archive["output"])
    pool = archive["output"] / "reviewer" / "pool.json"
    pool.write_text(pool.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ExperimentError, match="integrity check failed"):
        load_package(archive["output"])


def test_consistency_aliases_are_hidden_and_unlabelled(archive):
    report = json.loads(archive["report"].read_text())
    report["runs"].append(
        {
            **report["runs"][0],
            "query_id": "SECOND_PRIVATE_QUERY",
            "query": "Another question",
        }
    )
    judged = json.loads(archive["judged"].read_text())
    judged["queries"].append(
        {
            **judged["queries"][0],
            "query_id": "SECOND_PRIVATE_QUERY",
            "query": "Another question",
        }
    )
    write(archive["report"], report)
    write(archive["judged"], judged)
    record = json.loads((archive["run"] / "run.json").read_text())
    record["arms"][0]["measure"][0]["report_sha256"] = sha(archive["report"])
    record["judgments"][0]["kept_sha256"] = sha(archive["judged"])
    write(archive["run"] / "run.json", record)
    build_pool(archive["spec"], archive["output"])
    loaded = load_package(archive["output"])
    key = json.loads((archive["output"] / "private" / "key.json").read_text())
    assert len(key["repeats"]) == 1
    assert "repeat_of" not in json.dumps(loaded["pool"])
    result = check_annotations(
        archive["output"], archive["output"] / "reviewer" / "template.json"
    )
    assert result["status"] == "incomplete"


def test_annotation_cli_checks_pending_without_treating_it_as_negative(archive, capsys):
    from rag_experiments.cli import main

    assert (
        main(
            [
                "annotation",
                "build",
                "--spec",
                str(archive["spec"]),
                "--output",
                str(archive["output"]),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "pending_human_judgments"
    template = archive["output"] / "reviewer" / "template.json"
    assert (
        main(
            [
                "annotation",
                "check",
                "--pool",
                str(archive["output"]),
                "--judgments",
                str(template),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"
    assert (
        main(
            [
                "annotation",
                "check",
                "--pool",
                str(archive["output"]),
                "--judgments",
                str(template),
                "--require-complete",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["policy_ready"] is False
    assert main(["annotation", "inspect", "--pool", str(archive["output"])]) == 0
    assert json.loads(capsys.readouterr().out)["integrity"] == "verified"


def test_render_failure_leaves_no_partial_packet(archive, monkeypatch):
    from rag_experiments.annotation import build as module

    def fail(*args, **kwargs):
        raise RuntimeError("render interrupted")

    monkeypatch.setattr(module, "render_review", fail)
    with pytest.raises(RuntimeError, match="render interrupted"):
        build_pool(archive["spec"], archive["output"])
    assert not archive["output"].exists()
    assert not list(archive["output"].parent.glob(".annotation-*"))


def test_missing_integrity_entry_is_not_treated_as_optional(archive):
    build_pool(archive["spec"], archive["output"])
    manifest = archive["output"] / "private" / "manifest.json"
    data = json.loads(manifest.read_text())
    data["files"].pop("private/key.json")
    write(manifest, data)
    with pytest.raises(ExperimentError, match="omits required"):
        load_package(archive["output"])


def test_unrecorded_file_in_reviewer_packet_is_detected(archive):
    build_pool(archive["spec"], archive["output"])
    (archive["output"] / "reviewer" / "leaked-key.json").write_text("{}")
    with pytest.raises(ExperimentError, match="inventory changed"):
        load_package(archive["output"])


def test_unknown_selections_and_fields_are_refused(archive):
    data = json.loads(archive["spec"].read_text())
    data["inputs"][0]["arms"] = ["NOT_A_MEASURED_ARM"]
    write(archive["spec"], data)
    with pytest.raises(ExperimentError, match="Unknown selected arms"):
        build_pool(archive["spec"], archive["output"])
    data["unknown"] = True
    write(archive["spec"], data)
    with pytest.raises(ExperimentError, match="Unsupported"):
        build_pool(archive["spec"], archive["output"])


def test_symlinked_report_cannot_bypass_retained_hash(archive):
    path = archive["report"]
    saved = path.with_suffix(".original")
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(ExperimentError, match="symbolic link"):
        build_pool(archive["spec"], archive["output"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("target_id", "WRONG_TARGET"),
        ("target_chunk_id", "INTERNAL_COPY"),
        ("family_id", "UNDECLARED_FAMILY"),
    ],
)
def test_frozen_target_and_family_bindings_are_checked(archive, key, value):
    report = json.loads(archive["report"].read_text())
    report["runs"][0][key] = value
    write(archive["report"], report)
    record = json.loads((archive["run"] / "run.json").read_text())
    record["arms"][0]["measure"][0]["report_sha256"] = sha(archive["report"])
    write(archive["run"] / "run.json", record)
    with pytest.raises(ExperimentError, match="frozen"):
        build_pool(archive["spec"], archive["output"])


@pytest.mark.parametrize("value", [None, 42, ["note"], {"note": "value"}])
def test_nontext_notes_are_not_valid_human_annotation(archive, value):
    build_pool(archive["spec"], archive["output"])
    template = json.loads(
        (archive["output"] / "reviewer" / "template.json").read_text()
    )
    template["items"][0]["notes"] = value
    response = archive["output"].parent / "response.json"
    write(response, template)
    with pytest.raises(ExperimentError, match="notes must be text"):
        check_annotations(archive["output"], response)


def synthetic_judgments(archive):
    path = archive["output"] / "reviewer" / "template.json"
    data = json.loads(path.read_text())
    data["annotator"] = "synthetic test fixture, not real author judgments"
    data["rubric_acknowledged"] = True
    for row in data["items"]:
        row.update(relevance="direct", usability="usable", source_verified=True)
    for row in data["pairs"]:
        row["relation"] = "copy"
    response = archive["output"].parent / "synthetic-response.json"
    write(response, data)
    return response


def test_pending_packet_refuses_handoff_and_produces_no_file(archive):
    from rag_experiments.annotation.export import export_handoff

    build_pool(archive["spec"], archive["output"])
    output = archive["output"].parent / "handoff.json"
    with pytest.raises(ExperimentError, match="incomplete"):
        export_handoff(
            archive["output"], archive["output"] / "reviewer" / "template.json", output
        )
    assert not output.exists()


def test_completed_handoff_joins_only_real_rankings_to_primary_labels(archive):
    from rag_experiments.annotation.export import export_handoff

    build_pool(archive["spec"], archive["output"])
    output = archive["output"].parent / "handoff.json"
    summary = export_handoff(archive["output"], synthetic_judgments(archive), output)
    handoff = json.loads(output.read_text())
    assert handoff["protocol"] == "author_pool_v1"
    assert handoff["role"] == "exploratory"
    assert len(handoff["judgments"]) == 3
    assert len(handoff["conditions"]) == 1
    assert len(handoff["conditions"][0]["rankings"][0]["passage_ids"]) == 2
    assert handoff["label_validation"]["status"] == "complete"
    assert summary["policy_ready"] is False
    assert not any("passage" in row for row in handoff["judgments"])
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ExperimentError, match="new nonlinked"):
        export_handoff(archive["output"], synthetic_judgments(archive), output)


def test_uncertain_label_is_not_changed_into_zero_relevance(archive):
    from rag_experiments.annotation.export import export_handoff

    build_pool(archive["spec"], archive["output"])
    response = synthetic_judgments(archive)
    data = json.loads(response.read_text())
    data["items"][0]["relevance"] = "uncertain"
    write(response, data)
    output = archive["output"].parent / "handoff.json"
    with pytest.raises(ExperimentError, match="adjudication_required"):
        export_handoff(archive["output"], response, output)
    assert not output.exists()


def test_handoff_cannot_write_inside_frozen_inputs_or_original_project(archive):
    from rag_experiments.annotation.export import export_handoff

    build_pool(archive["spec"], archive["output"])
    response = synthetic_judgments(archive)
    for output in (
        archive["output"] / "handoff.json",
        archive["project"] / "handoff.json",
        archive["output"].parent / "live-original" / "handoff.json",
    ):
        with pytest.raises(ExperimentError):
            export_handoff(archive["output"], response, output)


def test_toolkit_delegates_synthetic_quality_scoring_to_app(archive, app_source):
    from rag_experiments.annotation.export import export_handoff
    from rag_experiments.annotation.scoring import score_handoff

    app = app_source
    if not (app / "scripts" / "evaluate_pooled.py").is_file():
        pytest.skip("App-owned pooled scorer is not present in this development layout")
    build_pool(archive["spec"], archive["output"])
    handoff = archive["output"].parent / "synthetic-handoff.json"
    export_handoff(archive["output"], synthetic_judgments(archive), handoff)
    report = archive["output"].parent / "synthetic-pooled-report.json"
    summary = score_handoff(handoff, app, report, bootstrap_samples=50)
    scored = json.loads(report.read_text())
    assert summary["policy_ready"] is False
    assert scored["report_meta"]["measurement_kind"] == "pooled_author_labels"
    assert scored["decision"]["policy_ready"] is False
    assert scored["no_answer"]["measured"] is False
    assert (
        scored["conditions"][0]["queries"][0]["direct_usable_precision_returned"] == 1.0
    )
    with pytest.raises(ExperimentError, match="new nonlinked"):
        score_handoff(handoff, app, report)


def test_overlapping_selected_partitions_cannot_duplicate_ranked_queries(archive):
    from rag_experiments.annotation.export import prepare_handoff

    record = json.loads((archive["run"] / "run.json").read_text())
    record["judgments"].append({**record["judgments"][0], "name": "overlap"})
    record["arms"][0]["measure"].append(
        {**record["arms"][0]["measure"][0], "split": "overlap"}
    )
    record["arms"][0]["reports"]["overlap"] = str(archive["report"])
    write(archive["run"] / "run.json", record)
    spec = json.loads(archive["spec"].read_text())
    spec["inputs"][0]["splits"].append("overlap")
    write(archive["spec"], spec)
    build_pool(archive["spec"], archive["output"])
    with pytest.raises(ExperimentError, match="partitions overlap"):
        prepare_handoff(archive["output"], synthetic_judgments(archive))


def test_missing_budget_is_named_at_export_not_hidden_in_app_refusal(archive):
    from rag_experiments.annotation.export import prepare_handoff

    report = json.loads(archive["report"].read_text())
    report["runs"][0].pop("candidate_depth")
    write(archive["report"], report)
    record = json.loads((archive["run"] / "run.json").read_text())
    record["arms"][0]["measure"][0]["report_sha256"] = sha(archive["report"])
    write(archive["run"] / "run.json", record)
    build_pool(archive["spec"], archive["output"])
    with pytest.raises(ExperimentError, match="observed budget candidate_depth"):
        prepare_handoff(archive["output"], synthetic_judgments(archive))


def test_relative_run_selection_survives_into_handoff(archive):
    from rag_experiments.annotation.export import prepare_handoff

    spec = json.loads(archive["spec"].read_text())
    spec["inputs"][0]["run"] = "retained-run"
    write(archive["spec"], spec)
    build_pool(archive["spec"], archive["output"])
    handoff = prepare_handoff(archive["output"], synthetic_judgments(archive))
    assert len(handoff["conditions"]) == 1
    assert handoff["conditions"][0]["metadata"]["partitions"] == ["exploratory"]
