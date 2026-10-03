#!/usr/bin/env python3
"""Create two exploratory query partitions without splitting target families.

No target is excluded by default. Exclusions require an ID and a reason.
Each partition declares only the targets its own queries use. The generated
manifest records the input digest, family allocation, and exclusion reasons.
Partitioning an already inspected benchmark does not create an untouched test set.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from rag_experiments._families import connected_families

PARTITIONS = ("exploratory-a", "exploratory-b")


def split(
    judged: dict[str, Any], exclusions: dict[str, str] | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Keep each target and each declared question family in one partition."""

    exclusions = dict(exclusions or {})
    targets = {str(t["target_id"]): t for t in judged["targets"]}
    if len(targets) != len(judged["targets"]):
        raise ValueError("Target IDs must be unique")
    queries = judged["queries"]
    if len({str(q["query_id"]) for q in queries}) != len(queries):
        raise ValueError("Query IDs must be unique")
    used = {str(q["target_id"]) for q in queries}
    unknown = (used | set(exclusions)) - set(targets)
    if unknown:
        raise ValueError(f"Unknown target IDs: {sorted(unknown)}")
    if any(not reason.strip() for reason in exclusions.values()):
        raise ValueError("Each exclusion requires a nonempty reason")

    usable = sorted(used - set(exclusions))
    selected_queries = [q for q in queries if str(q["target_id"]) in usable]
    family_map = connected_families(
        selected_queries, {key: targets[key] for key in usable}
    )
    declared_links = any(
        q.get("family_id") or targets[str(q["target_id"])].get("family_id")
        for q in selected_queries
    )
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for target in usable:
        groups[family_map[target]].add(target)
    if len(groups) < 2:
        raise ValueError("At least two independent target families are required")

    allocations = {name: set() for name in PARTITIONS}
    for position, group in enumerate(sorted(groups)):
        allocations[PARTITIONS[position % 2]].update(groups[group])
    outputs = []
    for name in PARTITIONS:
        selected = allocations[name]
        document = dict(judged)
        document["targets"] = [targets[key] for key in sorted(selected)]
        document["queries"] = sorted(
            (q for q in queries if str(q["target_id"]) in selected),
            key=lambda q: str(q["query_id"]),
        )
        document["protocol"] = (
            f"{judged.get('protocol', '')} Exploratory partition {name}; "
            "target and declared question families are disjoint. "
            "This partition is not a previously unseen holdout."
        ).strip()
        document["partition_metadata"] = {
            "role": "exploratory",
            "method": "sorted-family-alternation",
            "excluded_targets": exclusions,
            "unqueried_targets": sorted(set(targets) - used),
            "unqueried_reason": "No query references these input targets",
            "declared_family_links": declared_links,
            "target_groups": {
                group: sorted(members)
                for group, members in groups.items()
                if members <= selected
            },
        }
        outputs.append({"name": name, "document": copy.deepcopy(document)})
    return outputs[0], outputs[1], exclusions


def exclusion_map(entries: list[str]) -> dict[str, str]:
    """Parse explicit ID=reason exclusions; refuse duplicates and empty reasons."""

    exclusions = {}
    for entry in entries:
        target, separator, reason = entry.partition("=")
        if not separator or not target.strip() or not reason.strip():
            raise ValueError("--exclude-target requires TARGET_ID=reason")
        if target.strip() in exclusions:
            raise ValueError(f"Duplicate exclusion: {target.strip()}")
        exclusions[target.strip()] = reason.strip()
    return exclusions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("judgments"))
    parser.add_argument("--exclude-target", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        source = args.judged.read_bytes()
        first, second, exclusions = split(
            json.loads(source), exclusion_map(args.exclude_target)
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))

    destinations = [args.out / f"{name}-queries.json" for name in PARTITIONS]
    manifest_path = args.out / "partition-manifest.json"
    if any(path.exists() for path in [*destinations, manifest_path]):
        parser.error("Output already exists; use a new partition directory")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "role": "exploratory",
        "input_sha256": hashlib.sha256(source).hexdigest(),
        "excluded_targets": exclusions,
        "unqueried_targets": first["document"]["partition_metadata"][
            "unqueried_targets"
        ],
        "partitions": [],
    }
    for partition, path in zip((first, second), destinations, strict=True):
        document = partition["document"]
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        counts = collections.Counter(q["class"] for q in document["queries"])
        info = {
            "name": partition["name"],
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "target_ids": [t["target_id"] for t in document["targets"]],
            "query_count": len(document["queries"]),
            "classes": dict(counts),
            "target_groups": document["partition_metadata"]["target_groups"],
            "declared_family_links": document["partition_metadata"][
                "declared_family_links"
            ],
        }
        manifest["partitions"].append(info)
        print(
            f"{partition['name']}: {info['query_count']} queries, "
            f"{len(info['target_ids'])} targets; {path}"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Exclusions: {exclusions}; manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
