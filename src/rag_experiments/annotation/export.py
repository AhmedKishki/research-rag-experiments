"""Join completed author judgments to retained rankings without scoring them."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..errors import ExperimentError
from ..report.record import read_record
from ..sandbox.layout import refuse_nested
from .build import _private_output, load_package
from .inputs import digest, read_json, require_hash
from .schema import validate_annotations

HANDOFF_PROTOCOL = "author_pool_v1"


def _protected_roots(package: Path, manifest: dict[str, Any]) -> list[Path]:
    roots = {package.resolve()}
    snapshots = {entry["path"]: entry["sha256"] for entry in manifest["input_files"]}
    for provenance in manifest["provenance"]:
        run = Path(provenance["run"])
        require_hash(run / "run.json", snapshots.get(str(run / "run.json")))
        record = read_record(run)
        roots.update((run, Path(record["source_project"]["project_root"])))
        roots.update(Path(arm["sandbox"]["root"]) for arm in record["arms"])
        engine = record["engine"].get("before") or record["engine"]
        if engine.get("root"):
            roots.add(Path(engine["root"]))
    return sorted(roots)


def prepare_handoff(package: Path, judgments: Path) -> dict[str, Any]:
    loaded = load_package(package)
    manifest, pool = loaded["manifest"], loaded["pool"]
    annotation_sha = digest(judgments)
    annotation = read_json(judgments)
    validation = validate_annotations(
        pool, annotation, pool_sha256=loaded["pool_sha256"]
    )
    if validation["status"] != "complete":
        raise ExperimentError(
            f"Author judgments are {validation['status']}; supply completed, acknowledged, adjudicated labels. No scores were produced."
        )
    key = read_json(package / "private" / "key.json")
    rows = {r["item_id"]: r for r in annotation["items"]}
    conflicts = [
        alias
        for alias, primary in key["repeats"].items()
        if any(
            rows[alias][field] != rows[primary][field]
            for field in ("relevance", "usability", "source_verified")
        )
    ]
    if conflicts:
        raise ExperimentError(
            "Consistency-repeat judgments disagree; adjudicate before exporting"
        )
    qmap = key["question_map"]
    selected = {q["query_id"]: q for q in qmap.values()}
    bound: dict[tuple[str, str], str] = {}
    exported = []
    for item_id, binding in key["item_map"].items():
        qid, cid = binding["query_id"], binding["chunk_id"]
        span = {
            name: binding[name]
            for name in (
                "source_relative_path",
                "locator",
                "content_sha256",
                "chunk_id",
            )
        }
        identity = (
            "passage-"
            + hashlib.sha256(
                json.dumps(span, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
        )
        bound[(qid, cid)] = identity
        row = rows[item_id]
        exported.append(
            {
                "query_id": qid,
                "passage_id": identity,
                **span,
                **{
                    name: row[name]
                    for name in ("relevance", "usability", "source_verified", "notes")
                },
            }
        )
    relation_rows = {r["pair_id"]: r for r in annotation["pairs"]}
    relations = []
    for pair_id, binding in key["pair_map"].items():
        left, right = binding["chunk_ids"]
        qid = binding["query_id"]
        relations.append(
            {
                "query_id": qid,
                "left_passage_id": bound[(qid, left)],
                "right_passage_id": bound[(qid, right)],
                "relation": relation_rows[pair_id]["relation"],
                "notes": relation_rows[pair_id]["notes"],
            }
        )

    # Collapse-only selections contribute author review material, not a quality
    # condition whose complete returned list may never have entered the pool.
    selections = manifest["specification"]["inputs"]
    result_origins = {
        (str(Path(s["run"]).resolve()), arm, split, mode)
        for s in selections
        if s["include_results"]
        for arm in s["arms"]
        for split in s["splits"]
        for mode in s["modes"]
    }
    conditions = {}
    expected_conditions = {
        Path(run).name + "/" + arm + "/" + mode
        for run, arm, split, mode in result_origins
    }
    for provenance in manifest["provenance"]:
        run, arm, split = provenance["run"], provenance["arm"], provenance["split"]
        path = Path(provenance["report"])
        require_hash(path, provenance["report_sha256"])
        report = read_json(path)
        for row in report["runs"]:
            qid, mode = row["query_id"], row["mode"]
            if qid not in selected or (run, arm, split, mode) not in result_origins:
                continue
            condition_id = Path(run).name + "/" + arm + "/" + mode
            condition = conditions.setdefault(
                condition_id,
                {
                    "condition_id": condition_id,
                    "metadata": {
                        "run": run,
                        "arm": arm,
                        "mode": mode,
                        "retrieval_harness": report["harness"],
                        "partitions": [],
                    },
                    "rankings": [],
                },
            )
            if condition["metadata"]["run"] != run:
                raise ExperimentError(
                    f"Two retained run paths collide on condition {condition_id}; select unambiguous run identities"
                )
            if split not in condition["metadata"]["partitions"]:
                condition["metadata"]["partitions"].append(split)
            if any(r["query_id"] == qid for r in condition["rankings"]):
                raise ExperimentError(
                    f"Selected partitions overlap for {condition_id}, query {qid}; select disjoint partitions"
                )
            ids = row["returned_chunk_ids"]
            if any((qid, cid) not in bound for cid in ids):
                raise ExperimentError(
                    "A scored ranking contains a passage not graded in this pool"
                )
            _validate_budget(row, context=f"{path}: {arm}/{split}/{mode}, query {qid}")
            condition["rankings"].append(
                {
                    "query_id": qid,
                    "requested_k": row["top_k"],
                    "passage_ids": [bound[(qid, cid)] for cid in ids],
                    **{
                        name: row.get(name)
                        for name in (
                            "candidate_depth",
                            "rerank_window",
                            "rerank_requested",
                            "reranked",
                            "rerank_fallback",
                        )
                    },
                }
            )
    if not conditions:
        raise ExperimentError(
            "The packet has no fully pooled result conditions to score"
        )
    if set(conditions) != expected_conditions:
        raise ExperimentError(
            f"Selected scoring conditions produced no pilot rows: {sorted(expected_conditions - set(conditions))}"
        )
    for condition in conditions.values():
        if {r["query_id"] for r in condition["rankings"]} != set(selected):
            raise ExperimentError(
                "Selected conditions do not cover the same frozen question cohort"
            )
        condition["rankings"].sort(key=lambda r: r["query_id"])
    require_hash(judgments, annotation_sha)
    return {
        "schema_version": 1,
        "protocol": HANDOFF_PROTOCOL,
        "role": "exploratory",
        "provenance": {
            "pool_id": pool["pool_id"],
            "pool_sha256": loaded["pool_sha256"],
            "rubric_sha256": manifest["rubric_sha256"],
            "annotation_sha256": annotation_sha,
            "generation_id": manifest["generation_id"],
            "generation_chunks_sha256": manifest["generation_chunks_sha256"],
            "generation_manifest_sha256": manifest["generation_manifest_sha256"],
            "rubric": pool["rubric"],
            "protected_roots": [
                str(root) for root in _protected_roots(package, manifest)
            ],
            "exclusions": manifest["exclusions"],
            "retrieval_provenance": manifest["provenance"],
            "label_scope": "current generation occurrences; re-chunking requires new source-span adjudication",
        },
        "questions": [
            {"query_id": qid, "family_id": selected[qid]["family_id"]}
            for qid in sorted(selected)
        ],
        "judgments": sorted(exported, key=lambda r: (r["query_id"], r["passage_id"])),
        "relations": relations,
        "conditions": list(conditions.values()),
        "annotation_consistency": {
            "checked_items": len(key["repeats"]),
            "quality_conflicts": [],
        },
        "label_validation": {
            "status": "complete",
            "rubric_acknowledged": True,
            "annotator": annotation["annotator"],
        },
    }


def _validate_budget(row: dict[str, Any], *, context: str) -> None:
    for name in ("top_k", "candidate_depth", "rerank_window"):
        if type(row.get(name)) is not int or row[name] < (
            0 if name == "rerank_window" else 1
        ):
            raise ExperimentError(
                f"Missing or invalid observed budget {name}: {context}"
            )
    if (
        any(
            type(row.get(name)) is not bool for name in ("rerank_requested", "reranked")
        )
        or "rerank_fallback" not in row
    ):
        raise ExperimentError(f"Missing rerank evidence: {context}")
    if row["rerank_requested"] and (not row["reranked"] or row["rerank_fallback"]):
        raise ExperimentError(f"Reranking was not applied: {context}")


def export_handoff(package: Path, judgments: Path, output: Path) -> dict[str, Any]:
    package = package.expanduser().resolve()
    judgments = judgments.expanduser().absolute()
    output = output.expanduser().absolute()
    refuse_nested(output, package, what="judgment handoff", inside="frozen packet")
    loaded = load_package(package)
    for root in _protected_roots(package, loaded["manifest"]):
        refuse_nested(
            output,
            root,
            what="judgment handoff",
            inside="retained input or original project",
        )
    if (
        output == judgments
        or output.exists()
        or output.is_symlink()
        or any(p.is_symlink() for p in output.parents)
    ):
        raise ExperimentError(
            "Handoff output must be a new nonlinked path outside the packet and its judgments"
        )
    data = prepare_handoff(package, judgments)
    _private_output(output.parent / ".handoff-output")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".handoff-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # An exclusive link publishes only our new file; no original data is
        # hard-linked and no pre-existing handoff can be replaced.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "protocol": HANDOFF_PROTOCOL,
        "output": str(output),
        "sha256": digest(output),
        "questions": len(data["questions"]),
        "judgments": len(data["judgments"]),
        "conditions": len(data["conditions"]),
        "role": "exploratory",
        "policy_ready": False,
    }
