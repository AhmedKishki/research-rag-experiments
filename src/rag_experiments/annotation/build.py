"""Prepare private, arm-blinded author work without assigning any judgments."""

from __future__ import annotations

import copy
import hashlib
import hmac
import itertools
import json
import os
import random
import secrets
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ..errors import ExperimentError
from ..sandbox.layout import refuse_nested
from .inputs import (
    SEGMENT,
    beneath,
    digest,
    read_candidates,
    read_evidence,
    read_json,
    require_hash,
    safe_file,
    verify_inputs,
)
from .review import render_review
from .schema import (
    RUBRIC,
    canonical_digest,
    make_template,
    validate_annotations,
    validate_pool,
)

SPEC_KEYS = {
    "schema_version",
    "name",
    "inputs",
    "include_designated_targets",
    "max_candidates_per_query",
    "max_pairs_per_query",
    "repeat_fraction",
    "family_limit",
    "seed",
}
INPUT_KEYS = {
    "run",
    "arms",
    "splits",
    "modes",
    "include_results",
    "include_collapsed_pairs",
}


def _list(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(v, str) or not v for v in value)
    ):
        raise ExperimentError(f"{name} requires a nonempty string list")
    if len(value) != len(set(value)):
        raise ExperimentError(f"Duplicate {name} entries")
    return value


def load_spec(path: Path) -> dict[str, Any]:
    spec = read_json(path)
    if (
        set(spec) - SPEC_KEYS
        or spec.get("schema_version") != 1
        or isinstance(spec.get("schema_version"), bool)
    ):
        raise ExperimentError("Unsupported annotation specification fields or schema")
    if not isinstance(spec.get("name"), str) or not SEGMENT.fullmatch(spec["name"]):
        raise ExperimentError("Annotation name must be one safe segment")
    if not isinstance(spec.get("inputs"), list) or not spec["inputs"]:
        raise ExperimentError("Annotation inputs must be a nonempty list")
    for entry in spec["inputs"]:
        if (
            not isinstance(entry, dict)
            or set(entry) - INPUT_KEYS
            or not isinstance(entry.get("run"), str)
        ):
            raise ExperimentError("Invalid annotation input selection")
        for field in ("arms", "splits", "modes"):
            _list(entry.get(field), field)
        for field, default in (
            ("include_results", True),
            ("include_collapsed_pairs", True),
        ):
            entry.setdefault(field, default)
            if not isinstance(entry[field], bool):
                raise ExperimentError(f"{field} must be boolean")
    spec.setdefault("include_designated_targets", True)
    if not isinstance(spec["include_designated_targets"], bool):
        raise ExperimentError("include_designated_targets must be boolean")
    for key, default in (("max_candidates_per_query", 60), ("max_pairs_per_query", 6)):
        spec.setdefault(key, default)
        if type(spec[key]) is not int or not 1 <= spec[key] <= 500:
            raise ExperimentError(f"{key} must be between 1 and 500")
    spec.setdefault("repeat_fraction", 0.1)
    fraction = spec["repeat_fraction"]
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 <= fraction <= 0.25
    ):
        raise ExperimentError("repeat_fraction must be between zero and .25")
    if spec.get("family_limit") is not None and (
        type(spec["family_limit"]) is not int or spec["family_limit"] < 1
    ):
        raise ExperimentError("family_limit must be a positive integer or null")
    if spec.get("seed") is not None and (
        type(spec["seed"]) is not int or spec["seed"] < 0
    ):
        raise ExperimentError("seed must be a nonnegative integer or null")
    return spec


def _write(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    path.chmod(0o600)


def _private_output(output: Path) -> None:
    """Refuse generated passage data inside a non-ignored Git location."""

    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    try:
        repo = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return  # Outside-Git outputs remain owner-only.
    if repo.returncode == 0:
        for relative in (
            "reviewer/pool.json",
            "reviewer/originals/Oprobe.pdf",
            "private/key.json",
            "private/manifest.json",
        ):
            ignored = subprocess.run(
                [
                    "git",
                    "-C",
                    repo.stdout.strip(),
                    "check-ignore",
                    "-q",
                    "--",
                    str(output / relative),
                ],
                check=False,
            )
            if ignored.returncode:
                raise ExperimentError(
                    "Private annotation output/staging is not fully Git-ignored; use annotations/ or an external directory"
                )


def _output_guard(output: Path, evidence) -> None:
    if output.exists() or output.is_symlink():
        raise ExperimentError(f"Annotation package already exists: {output}")
    if any(p.is_symlink() for p in output.parents):
        raise ExperimentError("Annotation output cannot traverse a symbolic link")
    for root in evidence.roots | evidence.source_projects | evidence.engine_roots:
        refuse_nested(
            output,
            root,
            what="annotation output",
            inside="retained input or original project",
        )
    _private_output(output)


def _opaque(key: bytes, kind: str, value: str) -> str:
    return kind + "-" + hmac.new(key, value.encode(), hashlib.sha256).hexdigest()[:24]


def build_pool(spec_path: Path, output: Path) -> dict[str, Any]:
    """Publish one complete package atomically; leave no partial reviewer data."""

    spec_path = safe_file(spec_path)
    spec_sha = digest(spec_path)
    spec = load_spec(spec_path)
    output = output.expanduser().absolute()
    evidence = read_evidence(spec, spec_path.parent)
    _output_guard(output, evidence)
    chunks = read_candidates(evidence)
    seed = spec.get("seed")
    seed = secrets.randbits(256) if seed is None else seed
    rng = random.Random(seed)
    key = hashlib.sha256(str(seed).encode()).digest()
    families = sorted({q["family_id"] for q in evidence.queries.values()})
    if spec.get("family_limit"):
        rng.shuffle(families)
        families = families[: spec["family_limit"]]
    questions = {
        qid: q for qid, q in evidence.queries.items() if q["family_id"] in families
    }
    query_ids = sorted(questions)
    rng.shuffle(query_ids)
    pool = {
        "schema_version": 1,
        "pool_id": "pool-" + uuid.uuid4().hex,
        "role": "exploratory",
        "rubric": copy.deepcopy(RUBRIC),
        "questions": [],
        "items": [],
        "pairs": [],
    }
    private = {
        "seed": seed,
        "role": "exploratory",
        "item_map": {},
        "question_map": {},
        "pair_map": {},
        "repeats": {},
        "sources": {},
    }
    source_files = {}
    for qid in query_ids:
        opaque_q = _opaque(key, "Q", qid)
        question = questions[qid]
        pool["questions"].append({"question_id": opaque_q, "query": question["query"]})
        private["question_map"][opaque_q] = {"query_id": qid, **question}
        candidates = sorted(cid for query, cid in evidence.origins if query == qid)
        if not candidates:
            raise ExperimentError(
                f"Selected inputs produce no candidates for {qid}; this is not a no-answer judgment"
            )
        if len(candidates) > spec["max_candidates_per_query"]:
            raise ExperimentError(
                f"Mandatory union for {qid} exceeds candidate limit; select fewer arms or raise the limit"
            )
        rng.shuffle(candidates)
        ids = {}
        for cid in candidates:
            candidate = chunks[cid]
            source = candidate["_annotation_source"]
            original_key = (
                source["source_relative_path"] + "\0" + source["original_sha256"]
            )
            source_alias = (
                _opaque(key, "O", original_key) + source["original_path"].suffix.lower()
            )
            source_files[source_alias] = source["original_path"]
            private["sources"][source_alias] = {
                "source_relative_path": source["source_relative_path"],
                "sha256": source["original_sha256"],
            }
            opaque_i = _opaque(key, "I", qid + "\0" + cid)
            ids[cid] = opaque_i
            shown_source = {
                k: copy.deepcopy(v)
                for k, v in source.items()
                if k not in {"original_path", "original_sha256"}
            }
            shown_source["original"] = "originals/" + source_alias
            pool["items"].append(
                {
                    "item_id": opaque_i,
                    "question_id": opaque_q,
                    "passage": candidate["contents"],
                    "source": shown_source,
                }
            )
            private["item_map"][opaque_i] = {
                "query_id": qid,
                "chunk_id": cid,
                "document_id": candidate["document_id"],
                "source_id": candidate["source_id"],
                "source_relative_path": source["source_relative_path"],
                "locator": candidate["locator"],
                "content_sha256": hashlib.sha256(
                    candidate["contents"].encode()
                ).hexdigest(),
                "origins": evidence.origins[(qid, cid)],
            }
        mandatory = evidence.pairs.get(qid, {})
        if len(mandatory) > spec["max_pairs_per_query"]:
            raise ExperimentError(
                f"Mandatory source pairs for {qid} exceed the pair limit"
            )
        pairs = list(mandatory)
        sampled = [
            pair
            for pair in itertools.combinations(sorted(candidates), 2)
            if pair not in mandatory
        ]
        rng.shuffle(sampled)
        pairs.extend(sampled[: spec["max_pairs_per_query"] - len(pairs)])
        rng.shuffle(pairs)
        for left, right in pairs:
            if rng.choice((False, True)):
                left, right = right, left
            pair_id = _opaque(key, "P", qid + "\0" + "\0".join(sorted((left, right))))
            pool["pairs"].append(
                {
                    "pair_id": pair_id,
                    "question_id": opaque_q,
                    "left_item_id": ids[left],
                    "right_item_id": ids[right],
                }
            )
            private["pair_map"][pair_id] = {
                "query_id": qid,
                "chunk_ids": [left, right],
                "origins": mandatory.get(tuple(sorted((left, right))), []),
                "sampled": tuple(sorted((left, right))) not in mandatory,
            }
    repeated = rng.sample(
        pool["items"], int(len(pool["items"]) * spec["repeat_fraction"])
    )
    for position, item in enumerate(repeated):
        alias = copy.deepcopy(item)
        alias_id = _opaque(key, "I", f"consistency\0{position}\0{item['item_id']}")
        alias["item_id"] = alias_id
        pool["items"].append(alias)
        private["repeats"][alias_id] = item["item_id"]
    rng.shuffle(pool["items"])
    rng.shuffle(pool["pairs"])
    validate_pool(pool)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".annotation-", dir=output.parent))
    stage.chmod(0o700)
    claim = None
    try:
        _private_output(stage)
        reviewer, hidden = stage / "reviewer", stage / "private"
        for folder in (reviewer, hidden, reviewer / "originals"):
            folder.mkdir(mode=0o700)
        _write(reviewer / "pool.json", pool)
        pool_sha = digest(reviewer / "pool.json")
        _write(reviewer / "rubric.json", pool["rubric"])
        _write(reviewer / "template.json", make_template(pool, pool_sha256=pool_sha))
        html = reviewer / "review.html"
        html.write_text(render_review(pool, pool_sha256=pool_sha), encoding="utf-8")
        html.chmod(0o600)
        for alias, original in source_files.items():
            copied = reviewer / "originals" / alias
            shutil.copy2(original, copied)
            copied.chmod(0o600)
            if (copied.stat().st_dev, copied.stat().st_ino) == (
                original.stat().st_dev,
                original.stat().st_ino,
            ):
                raise ExperimentError("An annotation original shares its input inode")
            require_hash(copied, private["sources"][alias]["sha256"])
        _write(hidden / "key.json", private)
        files = {
            p.relative_to(stage).as_posix(): digest(p)
            for p in stage.rglob("*")
            if p.is_file()
        }
        manifest = {
            "schema_version": 1,
            "pool_id": pool["pool_id"],
            "role": "exploratory",
            "pool_sha256": pool_sha,
            "rubric_sha256": canonical_digest(pool["rubric"]),
            "name": spec["name"],
            "specification": spec,
            "specification_sha256": spec_sha,
            "provenance": evidence.provenance,
            "generation_id": evidence.generation_id,
            "generation_manifest_sha256": evidence.generation_identity[0],
            "generation_chunks_sha256": evidence.generation_identity[1],
            "input_files": [
                {"path": str(p), "sha256": sha}
                for p, sha in dict(evidence.snapshots).items()
            ],
            "files": files,
            "exclusions": evidence.exclusions,
            "incomplete_source_pairs": evidence.incomplete_pairs,
            "pair_sampling": {
                "mandatory": "retained collapse examples",
                "other": "uniform shuffled sample of remaining candidate pairs",
                "exhaustive": False,
            },
            "questions": len(pool["questions"]),
            "unique_items": len(pool["items"]) - len(repeated),
            "consistency_items": len(repeated),
            "pairs": len(pool["pairs"]),
            "families": len(families),
            "source_originals": len(source_files),
            "policy_ready": False,
            "notice": "Arm-blinded exploratory labels; the account owner can access the private mapping. No quality metric or label is inferred.",
        }
        _write(hidden / "manifest.json", manifest)
        verify_inputs(evidence)
        require_hash(spec_path, spec_sha)
        try:
            output.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise ExperimentError(
                "Annotation output appeared while building; nothing overwritten"
            ) from exc
        claim = (output.stat().st_dev, output.stat().st_ino)
        os.rename(stage / "reviewer", output / "reviewer")
        # Install the manifest last. Readers refuse an incomplete publication.
        os.rename(stage / "private", output / "private")
        stage.rmdir()
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        if (
            claim is not None
            and not output.is_symlink()
            and output.exists()
            and (output.stat().st_dev, output.stat().st_ino) == claim
        ):
            shutil.rmtree(output)
        raise
    return {
        "pool_id": pool["pool_id"],
        "role": "exploratory",
        "output": str(output),
        "review": str(output / "reviewer" / "review.html"),
        "template": str(output / "reviewer" / "template.json"),
        "questions": len(pool["questions"]),
        "items": len(pool["items"]),
        "pairs": len(pool["pairs"]),
        "source_originals": len(source_files),
        "status": "pending_human_judgments",
        "policy_ready": False,
    }


def load_package(root: Path) -> dict[str, Any]:
    root = root.expanduser().absolute()
    manifest = read_json(root / "private" / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("role") != "exploratory":
        raise ExperimentError("Unsupported annotation package manifest")
    required = {
        "reviewer/pool.json",
        "reviewer/rubric.json",
        "reviewer/template.json",
        "reviewer/review.html",
        "private/key.json",
    }
    if not isinstance(manifest.get("files"), dict) or not required <= set(
        manifest["files"]
    ):
        raise ExperimentError("Annotation manifest omits required integrity entries")
    found = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ExperimentError("Annotation package contains a symbolic link")
        if path.is_file():
            found.add(path.relative_to(root).as_posix())
    if found - {"private/manifest.json"} != set(manifest["files"]):
        raise ExperimentError(
            "Packet inventory changed; keep returned judgments outside the frozen packet"
        )
    for relative, expected in manifest["files"].items():
        require_hash(beneath(root, relative), expected)
    pool = read_json(root / "reviewer" / "pool.json")
    validate_pool(pool)
    if (
        pool["pool_id"] != manifest["pool_id"]
        or canonical_digest(pool["rubric"]) != manifest["rubric_sha256"]
    ):
        raise ExperimentError("Pool identity or rubric changed")
    require_hash(root / "reviewer" / "pool.json", manifest["pool_sha256"])
    for item in pool["items"]:
        if "reviewer/" + item["source"]["original"] not in manifest["files"]:
            raise ExperimentError("Annotation manifest does not bind a quoted original")
    key = read_json(root / "private" / "key.json")
    items = {item["item_id"]: item for item in pool["items"]}
    primary = set(key.get("item_map", {}))
    repeats = key.get("repeats", {})
    if (
        not isinstance(repeats, dict)
        or primary & set(repeats)
        or primary | set(repeats) != set(items)
    ):
        raise ExperimentError(
            "Private item/repeat maps do not cover the blinded inventory"
        )
    for alias, original in repeats.items():
        if original not in primary or {
            k: v for k, v in items[alias].items() if k != "item_id"
        } != {k: v for k, v in items[original].items() if k != "item_id"}:
            raise ExperimentError(
                "A consistency alias is not bound to its original card"
            )
    if set(key.get("pair_map", {})) != {
        pair["pair_id"] for pair in pool["pairs"]
    } or set(key.get("question_map", {})) != {
        question["question_id"] for question in pool["questions"]
    }:
        raise ExperimentError(
            "Private question/pair maps do not cover the blinded inventory"
        )
    if manifest.get("unique_items") != len(primary) or manifest.get(
        "consistency_items"
    ) != len(repeats):
        raise ExperimentError(
            "Annotation inventory counts disagree with their private maps"
        )
    return {"pool": pool, "manifest": manifest, "pool_sha256": manifest["pool_sha256"]}


def check_annotations(root: Path, annotation_path: Path) -> dict[str, Any]:
    loaded = load_package(root)
    return validate_annotations(
        loaded["pool"], read_json(annotation_path), pool_sha256=loaded["pool_sha256"]
    )
