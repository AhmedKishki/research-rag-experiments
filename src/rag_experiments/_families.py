"""Group targets connected by frozen target-level or question-level declarations."""

from typing import Any


def connected_families(
    queries: list[dict[str, Any]], targets: dict[str, dict[str, Any]]
) -> dict[str, str]:
    parent = {target: target for target in targets}
    owners = {}

    def root(target):
        while parent[target] != target:
            target = parent[target]
        return target

    for query in queries:
        target = str(query["target_id"])
        if target not in targets:
            raise ValueError(
                "A question references a target absent from its frozen input"
            )
        for label in (targets[target].get("family_id"), query.get("family_id")):
            if label is None or label == "":
                continue
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Family declarations must be nonempty strings")
            label = label.strip()
            if label in owners:
                first, second = root(owners[label]), root(target)
                parent[max(first, second)] = min(first, second)
            else:
                owners[label] = target
    return {target: root(target) for target in targets}
