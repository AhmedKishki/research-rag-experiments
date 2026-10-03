"""Read verified retained evidence without searching or reading a live corpus."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._families import connected_families
from .._imports import without_bytecode
from ..errors import ExperimentError
from ..report.record import read_record
from ..sandbox.layout import read_layout

SEGMENT = re.compile(r"[A-Za-z0-9_-]+\Z")
BIB_FIELDS = ("title", "authors", "year", "doi")


def safe_file(path: Path) -> Path:
    """Refuse file and directory links before resolving a retained input."""

    path = path.expanduser().absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ExperimentError(f"Annotation input is a symbolic link: {path}")
    if not path.is_file():
        raise ExperimentError(f"Retained annotation input is unavailable: {path}")
    return path.resolve()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with safe_file(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(safe_file(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperimentError(f"Cannot read annotation input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"Annotation input is not an object: {path}")
    return value


def beneath(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ExperimentError(f"Unsafe retained artifact path: {relative!r}")
    path = root / candidate
    if not path.resolve().is_relative_to(root.resolve()):
        raise ExperimentError(f"Artifact escapes its retained root: {path}")
    return safe_file(path)


def require_hash(path: Path, expected: Any) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ExperimentError(f"No usable retained hash for {path}")
    actual = digest(path)
    if actual != expected:
        raise ExperimentError(f"Retained input integrity check failed: {path}")
    return actual


@dataclass
class Evidence:
    queries: dict[str, dict[str, Any]] = field(default_factory=dict)
    origins: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    pairs: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    snapshots: list[tuple[Path, str]] = field(default_factory=list)
    roots: set[Path] = field(default_factory=set)
    source_projects: set[Path] = field(default_factory=set)
    engine_roots: set[Path] = field(default_factory=set)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    exclusions: dict[str, Any] = field(default_factory=dict)
    sandbox: Path | None = None
    generation_id: str | None = None
    generation_identity: tuple[str, str] | None = None
    incomplete_pairs: bool = False


@without_bytecode
def generation_snapshot(sandbox: Path, generation_id: str) -> dict[str, Any]:
    """Bind canonical text to the generation's immutable artifact lookup."""

    from research_rag.retrieval.artifact_lookup import LOOKUP_RELATIVE_PATH

    if not SEGMENT.fullmatch(generation_id):
        raise ExperimentError("Unsafe measured generation identifier")
    layout = read_layout(sandbox)
    root = layout.generation_root(generation_id)
    manifest_path = safe_file(root / "manifest.json")
    manifest = read_json(manifest_path)
    if manifest.get("generation_id") != generation_id:
        raise ExperimentError("Measured generation and retained manifest disagree")
    chunks_path = beneath(root, manifest.get("files", {}).get("chunks", ""))
    lookup_path = safe_file(root / LOOKUP_RELATIVE_PATH)
    try:
        with sqlite3.connect(
            lookup_path.as_uri() + "?mode=ro&immutable=1", uri=True
        ) as db:
            metadata = dict(db.execute("SELECT key, value FROM metadata"))
    except sqlite3.Error as exc:
        raise ExperimentError(
            f"Cannot read immutable artifact identity: {exc}"
        ) from exc
    chunks_sha = require_hash(chunks_path, metadata.get("chunks_sha256"))
    return {
        "layout": layout,
        "root": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": digest(manifest_path),
        "chunks_path": chunks_path,
        "chunks_sha256": chunks_sha,
        "lookup_path": lookup_path,
        "lookup_sha256": digest(lookup_path),
        "expected_count": int(metadata.get("chunks_count", -1)),
    }


def read_evidence(spec: dict[str, Any], base: Path) -> Evidence:
    result = Evidence()
    for selection in spec["inputs"]:
        run_root = (base / Path(selection["run"]).expanduser()).absolute()
        safe_file(run_root / "run.json")
        run_root = run_root.resolve()
        record = read_record(run_root)
        if (
            record.get("verdict") != "verified"
            or record.get("run", {}).get("kind") != "measurement"
            or not record.get("run", {}).get("complete")
            or record.get("source_project", {}).get("guard", {}).get("state")
            != "unchanged"
            or record.get("engine", {}).get("changed") is not False
        ):
            raise ExperimentError(
                f"A complete verified measurement is required: {run_root}"
            )
        result.roots.add(run_root)
        result.source_projects.add(
            Path(record["source_project"]["project_root"]).resolve()
        )
        engine = record.get("engine", {})
        engine_before = engine.get("before") or engine
        if engine_before.get("root"):
            result.engine_roots.add(Path(engine_before["root"]).resolve())
        result.snapshots.append((run_root / "run.json", digest(run_root / "run.json")))
        arms = {a["arm"]["name"]: a for a in record["arms"]}
        if set(selection["arms"]) - set(arms):
            raise ExperimentError(f"Unknown selected arms in {run_root}")
        for name in selection["arms"]:
            arm = arms[name]
            sandbox = Path(arm["sandbox"]["root"])
            if any(
                sandbox.resolve().is_relative_to(root)
                for root in result.source_projects | result.engine_roots
            ):
                raise ExperimentError(
                    "A retained sandbox must not name a live project or engine"
                )
            result.roots.add(sandbox.resolve())
            measures = {m["split"]: m for m in arm["measure"]}
            judged_inputs = {j["name"]: j for j in record["judgments"]}
            for split in selection["splits"]:
                if split not in measures or split not in judged_inputs:
                    raise ExperimentError(f"Unknown selected partition {split!r}")
                measure = measures[split]
                if measure.get("exit_code") != 0:
                    raise ExperimentError(
                        "A selected measurement did not finish successfully"
                    )
                path = Path(measure["report"])
                if not path.resolve().is_relative_to(run_root) or arm.get(
                    "reports", {}
                ).get(split) != str(path):
                    raise ExperimentError(
                        "Selected report is not owned by its retained run"
                    )
                report_sha = require_hash(path, measure.get("report_sha256"))
                report = read_json(path)
                if report.get("schema_version") != 3 or report.get("degraded"):
                    raise ExperimentError("An undegraded schema-3 report is required")
                kept = judged_inputs[split]
                judged_path = Path(kept["kept_at"])
                if not judged_path.resolve().is_relative_to(run_root):
                    raise ExperimentError(
                        "Frozen judged inputs are not owned by their retained run"
                    )
                judged_sha = require_hash(judged_path, kept.get("kept_sha256"))
                judged = read_json(judged_path)
                exclusions = judged.get("partition_metadata", {}).get(
                    "excluded_targets", {}
                )
                if any(
                    key in result.exclusions and result.exclusions[key] != value
                    for key, value in exclusions.items()
                ):
                    raise ExperimentError(
                        "Selected inputs disagree about an exclusion reason"
                    )
                result.exclusions.update(exclusions)
                query_specs = {str(q["query_id"]): q for q in judged["queries"]}
                target_specs = {str(t["target_id"]): t for t in judged["targets"]}
                frozen_families = connected_families(judged["queries"], target_specs)
                resolved_targets = {
                    str(t["target_id"]): t for t in report.get("targets", [])
                }
                generation = str(report["project"]["generation_id"])
                snapshot = generation_snapshot(sandbox, generation)
                identity = (snapshot["manifest_sha256"], snapshot["chunks_sha256"])
                if (
                    result.generation_identity is not None
                    and identity != result.generation_identity
                ):
                    raise ExperimentError(
                        "Selected inputs use different generation/text identities"
                    )
                result.generation_identity = identity
                result.generation_id = generation
                result.sandbox = result.sandbox or sandbox
                for artifact in ("manifest", "chunks", "lookup"):
                    result.snapshots.append(
                        (snapshot[f"{artifact}_path"], snapshot[f"{artifact}_sha256"])
                    )
                portable = snapshot["layout"].portable_root
                for filename in (
                    "source-catalog.json",
                    "source-metadata.json",
                    "source-exclusions.json",
                ):
                    file = portable / filename
                    if file.exists():
                        signature = digest(file)
                        previous = next(
                            (sha for p, sha in result.snapshots if p.name == filename),
                            None,
                        )
                        if previous is not None and signature != previous:
                            raise ExperimentError(
                                f"Selected review snapshots differ: {filename}"
                            )
                        result.snapshots.append((file, signature))
                modes_seen = {r["mode"] for r in report["runs"]}
                if set(selection["modes"]) - modes_seen:
                    raise ExperimentError("A selected mode was not measured")
                for row in report["runs"]:
                    if row["mode"] not in selection["modes"]:
                        continue
                    _consume_row(
                        result,
                        row,
                        selection,
                        query_specs,
                        frozen_families=frozen_families,
                        resolved_targets=resolved_targets,
                        include_target=spec["include_designated_targets"],
                        origin={"run": str(run_root), "arm": name, "split": split},
                    )
                result.provenance.append(
                    {
                        "run": str(run_root),
                        "arm": name,
                        "split": split,
                        "report": str(path),
                        "report_sha256": report_sha,
                        "judged": str(judged_path),
                        "judged_sha256": judged_sha,
                        "harness": report["harness"],
                        "generation_id": generation,
                    }
                )
                result.snapshots.extend([(path, report_sha), (judged_path, judged_sha)])
    if not result.queries or not result.origins:
        raise ExperimentError("The selected evidence produces an empty annotation pool")
    return result


def _consume_row(
    result: Evidence,
    row: dict[str, Any],
    selection: dict[str, Any],
    query_specs: dict[str, Any],
    *,
    frozen_families: dict[str, str],
    resolved_targets: dict[str, Any],
    include_target: bool,
    origin: dict[str, Any],
) -> None:
    if "rerank" in row["mode"] and (
        row.get("reranked") is not True or row.get("rerank_fallback")
    ):
        raise ExperimentError("A selected reranked row did not apply reranking")
    qid = str(row["query_id"])
    if qid not in query_specs:
        raise ExperimentError("Report query is absent from its frozen input")
    query = query_specs[qid]
    if row.get("query") != query["query"]:
        raise ExperimentError("Report question differs from its frozen input")
    target_id = str(query["target_id"])
    resolved = resolved_targets.get(target_id)
    if (
        row.get("target_id") != target_id
        or not resolved
        or resolved.get("chunk_id") != row["target_chunk_id"]
    ):
        raise ExperimentError(
            "Report designated target is not bound to the frozen question"
        )
    if row.get("family_id") != frozen_families[target_id]:
        raise ExperimentError("Report family differs from frozen declarations")
    facts = {
        "query": query["query"],
        "target_id": query["target_id"],
        "family_id": frozen_families[target_id],
        "target_chunk_id": row["target_chunk_id"],
    }
    if qid in result.queries and facts != result.queries[qid]:
        raise ExperimentError(
            "Selected reports disagree about a question or designated target"
        )
    result.queries[qid] = facts
    origin = {**origin, "mode": row["mode"], "query_id": qid}
    if selection["include_results"]:
        for rank, cid in enumerate(row["returned_chunk_ids"], 1):
            result.origins.setdefault((qid, cid), []).append({**origin, "rank": rank})
    if include_target:
        result.origins.setdefault((qid, row["target_chunk_id"]), []).append(
            {**origin, "designated_target": True}
        )
    if selection["include_collapsed_pairs"]:
        result.incomplete_pairs |= bool(row.get("collapsed_pairs_truncated"))
        for pair in row.get("collapsed_pairs", []):
            left, right = pair["chunk_id"], pair["repeated_chunk_id"]
            if left == right:
                raise ExperimentError("A collapse pair references one candidate twice")
            for cid in (left, right):
                result.origins.setdefault((qid, cid), []).append(
                    {**origin, "collapse_endpoint": True}
                )
            result.pairs.setdefault(qid, {}).setdefault(
                tuple(sorted((left, right))), []
            ).append({**origin, "collapse": pair})


def read_candidates(evidence: Evidence) -> dict[str, dict[str, Any]]:
    snapshot = generation_snapshot(evidence.sandbox, evidence.generation_id)
    layout, manifest = snapshot["layout"], snapshot["manifest"]
    needed = {cid for _, cid in evidence.origins}
    chunks = {}
    count = 0
    with snapshot["chunks_path"].open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
                chunk = json.loads(line)
                cid = chunk.get("chunk_id")
                if cid in needed:
                    if cid in chunks:
                        raise ExperimentError(
                            f"Duplicate canonical chunk identifier: {cid}"
                        )
                    chunks[cid] = chunk
    if count != snapshot["expected_count"] or needed - set(chunks):
        raise ExperimentError(
            "Canonical text coverage does not match the retained artifact identity"
        )
    documents = {d["document_id"]: d for d in manifest["documents"]}
    files = {f["source_id"]: f for f in manifest["source_files"]}
    catalog = read_json(layout.portable_root / "source-catalog.json")["sources"]
    reviews_path = layout.portable_root / "source-metadata.json"
    reviews = (
        read_json(reviews_path).get("sources", {}) if reviews_path.exists() else {}
    )
    original_hashes = {}
    for cid, chunk in chunks.items():
        doc = documents.get(chunk.get("document_id"))
        source = files.get(chunk.get("source_id"))
        if not doc or not source or doc.get("source_id") != chunk.get("source_id"):
            raise ExperimentError(f"Incomplete source identity for {cid}")
        relative = source["source_relative_path"]
        if catalog.get(chunk["source_id"], {}).get("source_relative_path") != relative:
            raise ExperimentError("Catalog and manifest source paths disagree")
        original = beneath(layout.source_root, relative)
        if original not in original_hashes:
            original_hashes[original] = require_hash(original, source["sha256"])
        original_sha = original_hashes[original]
        if original_sha != source["sha256"]:
            raise ExperimentError("One source path declares inconsistent hashes")
        if original.suffix.lower() not in {".pdf", ".epub"}:
            raise ExperimentError("An annotation original must be a PDF or EPUB")
        evidence.snapshots.append((original, original_sha))
        text = chunk.get("contents")
        locator = chunk.get("locator")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(locator, dict)
            or not locator
        ):
            raise ExperimentError(f"Missing canonical text or locator: {cid}")
        bibliography = {key: doc[key] for key in BIB_FIELDS if doc.get(key) is not None}
        for key, value in reviews.get(relative, {}).items():
            if key in BIB_FIELDS:
                if value is None:
                    bibliography.pop(key, None)
                else:
                    bibliography[key] = value
        authors = bibliography.get("authors")
        if isinstance(authors, list):
            if any(not isinstance(a, str) or not a.strip() for a in authors):
                raise ExperimentError("Invalid retained bibliographic author list")
            if authors:
                bibliography["authors"] = "; ".join(authors)
            else:
                bibliography.pop("authors", None)
        if type(bibliography.get("year")) is int:
            bibliography["year"] = str(bibliography["year"])
        bibliography = {
            key: value
            for key, value in bibliography.items()
            if not isinstance(value, str) or value.strip()
        }
        chunk["_annotation_source"] = {
            **bibliography,
            "source_relative_path": relative,
            "locator": locator,
            "original_path": original,
            "original_sha256": original_sha,
        }
    return chunks


def verify_inputs(evidence: Evidence) -> None:
    for path, expected in dict(evidence.snapshots).items():
        require_hash(path, expected)
