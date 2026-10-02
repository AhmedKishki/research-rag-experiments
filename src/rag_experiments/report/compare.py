"""Read a run's arms side by side, without computing a number.

Every column is read out of a report the app's own harness wrote. The comparison
adds nothing: it does not average, weight, rank, or decide. A column this module
cannot find is printed as a dash, because a metric the app's harness stopped
reporting is a fact about the harness and printing a zero for it would be a
different one.

Three rules come from what an ablation is allowed to claim:

- A run's first arm is the row every other arm is read against, so a difference is
  a difference from the arm the specification listed first rather than from a
  number chosen after the fact.
- One block per split and per mode, because a split is a different set of
  questions and a mode is a different question asked the same questions. Mixing
  them would hide which question a number answered.
- The candidate window and rerank budget are printed, and arms measured at unequal
  budgets are labelled as such. Comparing a wider candidate pool against a
  narrower one is a legitimate thing to want to see and is not an ablation; a
  table that presented the difference as one would be claiming a quality result
  from a budget change.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..errors import ExperimentError

#: The columns, in the order they are printed. Each is the label a reader sees,
#: the key it is read from in a row, and what that column is. A key a row does not
#: carry prints as a dash, because a metric the app's report stopped carrying is a
#: fact about the report and printing a zero for it would be a different one.
COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("arm", "arm", "The arm's own name, from the record."),
    ("n", "n", "Judged queries the arm actually ran in this split."),
    ("succ@1", "success_at_1", "The target passage ranked first."),
    ("succ@3", "success_at_3", "The target passage in the first three."),
    ("succ@k", "success_at_k", "The target passage inside the depth shown."),
    ("MRR", "mrr", "Mean reciprocal rank of the target passage."),
    ("nDCG", "ndcg_at_k", "Normalized discounted cumulative gain at that depth."),
    ("doc@k", "document_success_at_k", "The target's document retrieved at all."),
    ("spans", "mean_distinct_evidence_spans", "Different passages a result list held."),
    (
        "dup",
        "mean_exact_duplicate_slots",
        "Slots repeating another returned passage verbatim.",
    ),
    (
        "near",
        "mean_near_duplicate_slots",
        "Slots nearly wholly inside another returned passage.",
    ),
    (
        "1src",
        "mean_same_source_pairs",
        "Slots sharing a source file with another slot.",
    ),
    ("srcs", "mean_distinct_sources", "Distinct sources those passages came from."),
    (
        "rep%",
        "repeated_slot_rate",
        "Slots held by a passage more than one query returned.",
    ),
    ("withheld", "mean_withheld", "Candidates a relevance gate rejected."),
    ("p50 s", "p50_seconds", "The median measured query, in seconds."),
    ("p95 s", "p95_seconds", "The slowest measured query, in seconds."),
)

#: The settings that decide how much work a query was allowed, and the label each
#: is printed under. Read from the arm's pinned settings, because that is the
#: authority for what the engine was told; a budget the table cannot state is a
#: comparison a reader cannot judge.
BUDGET_SETTINGS: tuple[tuple[str, str], ...] = (
    ("retrieval.maximum_candidates", "cand"),
    ("retrieval.rerank_max_candidates", "rrank"),
)

#: What a column reads when the report does not carry it. A dash rather than a
#: zero, because a missing metric and a zero metric are not the same fact.
ABSENT = "-"

#: How a percentage is printed, so a column of means reads as one column.
PERCENT_COLUMNS = frozenset({"succ@1", "succ@3", "succ@k", "doc@k"})

#: How many columns wide each column is, except the arm name which takes the rest.
WIDTHS: dict[str, int] = {
    "arm": 0,
    "n": 4,
    "succ@1": 7,
    "succ@3": 7,
    "succ@k": 7,
    "MRR": 6,
    "nDCG": 6,
    "doc@k": 7,
    "spans": 6,
    "dup": 5,
    "near": 5,
    "1src": 5,
    "srcs": 5,
    "rep%": 6,
    "withheld": 9,
    "p50 s": 7,
    "p95 s": 7,
}

#: The quantiles a column shows for latency. Both are nearest-rank, so both are
#: values that were measured rather than an interpolation between two.
QUANTILES: tuple[tuple[str, str, float], ...] = (
    ("p50_seconds", "elapsed_seconds", 0.50),
    ("p95_seconds", "elapsed_seconds", 0.95),
)


def rows_for_run(
    record: dict[str, Any], mode: str, split: str | None = None
) -> list[dict[str, Any]]:
    """One row per arm of a run, for one measured mode and one split.

    A mode or a split the record has no row for is an empty table rather than an
    error: a specification that asked for one mode out of four is a run that
    measured one mode, and the record says which.
    """

    return [_row(arm, mode, split) for arm in record.get("arms") or []]


def _row(arm: dict[str, Any], mode: str, split: str | None) -> dict[str, Any]:
    """One arm's numbers for one mode and split, keyed as `COLUMNS` reads them.

    A key here and a key in `COLUMNS` that drift apart is a column that silently
    prints as a dash, so the row is built with the column table's own names.
    """

    report = _read(_report_path(arm, split))
    measured = ((report or {}).get("summary") or {}).get(mode) or {}
    overall = measured.get("overall") or {}
    per_query = [
        run for run in (report or {}).get("runs") or [] if str(run.get("mode")) == mode
    ]
    values = (arm.get("settings") or {}).get("values") or {}
    row: dict[str, Any] = {
        "arm": str((arm.get("arm") or {}).get("name") or "?"),
        "kind": str((arm.get("arm") or {}).get("kind") or ""),
        "n": overall.get("query_count"),
        "success_at_1": overall.get("success_at_1"),
        "success_at_3": overall.get("success_at_3"),
        "success_at_k": overall.get("success_at_k"),
        "mrr": overall.get("mrr"),
        "ndcg_at_k": overall.get("ndcg_at_k"),
        "document_success_at_k": overall.get("document_success_at_k"),
        "mean_distinct_sources": overall.get("mean_distinct_sources"),
        "mean_withheld": overall.get("mean_withheld"),
        "mean_distinct_evidence_spans": overall.get("mean_distinct_evidence_spans"),
        "mean_exact_duplicate_slots": overall.get("mean_exact_duplicate_slots"),
        "mean_near_duplicate_slots": overall.get("mean_near_duplicate_slots"),
        "mean_same_source_pairs": overall.get("mean_same_source_pairs"),
        # Measured across the mode's queries rather than within one query, so it
        # is reported per mode beside the per-query means rather than as one.
        "repeated_slot_rate": (report or {})
        .get("summary", {})
        .get(mode, {})
        .get("repeated_slot_rate"),
        "budgets": {key: values.get(key) for key, _label in BUDGET_SETTINGS},
        "present": bool(measured),
    }
    for name, key, quantile in QUANTILES:
        row[name] = _quantile(per_query, key, quantile)
    return row


def render_comparison(
    record: dict[str, Any],
    modes: tuple[str, ...] = (),
    split: str | None = None,
) -> str:
    """The table a reader compares arms with, as text.

    One block per split and per mode. With no split named, every split the run
    reported is tabulated, because a held-out number whose development counterpart
    is not shown cannot be read.
    """

    splits = (split,) if split else _splits(record)
    wanted = modes or tuple(
        dict.fromkeys(
            str(mode)
            for arm in record.get("arms") or []
            for mode in _modes_of(arm, splits)
        )
    )
    # The verdict leads whatever else is printed, including when there is nothing
    # else to print: a run with no readable number is exactly when a reader needs
    # to be told whether the corpus it measured is the corpus on disk.
    header = _verdict_line(record, str(record.get("verdict") or ""))
    if not splits or not wanted:
        return f"{header}\n\nNo arm of this run reported a measured mode."

    blocks: list[str] = []
    for one in splits:
        for mode in wanted:
            rows = rows_for_run(record, mode, one)
            if not any(row["present"] for row in rows):
                continue
            blocks.append(_table(one, mode, rows))
    if not blocks:
        return (
            f"{header}\n\n"
            f"No arm of this run reported any of: {', '.join(wanted)}"
            + (f" for split {splits[0]}" if split else "")
            + ".\nThe reports each arm wrote do not carry them."
        )
    return "\n".join([header, ""] + ["\n\n".join(blocks)])


def _table(split: str, mode: str, rows: list[dict[str, Any]]) -> str:
    names = [row["arm"] for row in rows]
    arm_width = max([len(name) for name in names] + [3])
    lines = [f"split: {split}    mode: {mode}"]
    lines.append(_budget_line(rows))
    lines.append(_header(arm_width))
    lines.append("-" * len(lines[-1]))
    for row in rows:
        lines.append(_line(row, arm_width))
    baseline = rows[0]
    if len(rows) > 1:
        lines.append("")
        equal = _equal_budgets(rows)
        lines.append(
            f"differences from {baseline['arm']}, the first arm the specification "
            f"listed:"
        )
        if not equal:
            # A factual statement of the confound, not a judgement about the run. A
            # candidate-window sweep varies the window on purpose, and a policy
            # ablation is only comparable at a fixed one; both are honest shapes,
            # and the reader is the one who knows which was intended.
            lines.append(
                "  the candidate window or rerank budget differs across these arms, "
                "so a difference here is a difference in budget as well as in "
                "policy; an ablation of a policy holds them equal."
            )
        for row in rows[1:]:
            lines.append(_delta_line(baseline, row, arm_width))
    return "\n".join(lines)


def _budget_line(rows: list[dict[str, Any]]) -> str:
    """State each arm's candidate window and rerank budget above the table.

    The settings live in the run record's per-arm pinned values, not in the table
    code: which settings decide the budget is the engine's fact, and this module
    names the ones it prints rather than deciding which ones matter.
    """

    named = ", ".join(label for _key, label in BUDGET_SETTINGS)
    parts: list[str] = []
    for row in rows:
        budgets = row.get("budgets") or {}
        stated = " ".join(
            f"{label}={ABSENT if budgets.get(key) is None else budgets.get(key)}"
            for key, label in BUDGET_SETTINGS
        )
        parts.append(f"{row['arm']}: {stated}")
    return f"budgets ({named}):   " + "   ".join(parts)


def _equal_budgets(rows: list[dict[str, Any]]) -> bool:
    """Whether every arm was given the same candidate window and rerank budget."""

    seen = set()
    for row in rows:
        budgets = row.get("budgets") or {}
        seen.add(tuple(budgets.get(key) for key, _label in BUDGET_SETTINGS))
    return len(seen) <= 1


def _header(arm_width: int) -> str:
    cells = [f"{'arm':<{arm_width}}"]
    for name, _key, _doc in COLUMNS[1:]:
        cells.append(f"{name:>{WIDTHS[name]}}")
    return "".join(cells)


def _line(row: dict[str, Any], arm_width: int) -> str:
    cells = [f"{row['arm']:<{arm_width}}"]
    for name, key, _doc in COLUMNS[1:]:
        cells.append(f"{_format(row.get(key), name):>{WIDTHS[name]}}")
    return "".join(cells)


def _delta_line(baseline: dict[str, Any], row: dict[str, Any], arm_width: int) -> str:
    cells = [f"{row['arm']:<{arm_width}}"]
    for name, key, _doc in COLUMNS[1:]:
        difference = _difference(baseline.get(key), row.get(key))
        cells.append(f"{_format(difference, name, delta=True):>{WIDTHS[name]}}")
    return "".join(cells)


def _format(value: Any, column: str, *, delta: bool = False) -> str:
    """Render one cell.

    A difference row carries an explicit sign, because a column of small numbers
    is unreadable when `-2.0` and `2.0` differ only by a character a reader has to
    hunt for. A negative zero is rendered as zero, since it is the arithmetic
    residue of two rounded equal values and reads as a movement that is not there.
    """

    if value is None:
        return ABSENT
    number = float(value)
    if number == 0.0:
        number = 0.0
    sign = "+" if delta and number > 0.0 else ""
    if column in PERCENT_COLUMNS or column == "rep%":
        return f"{sign}{100.0 * number:5.1f}%"
    if column in {"p50 s", "p95 s"}:
        return f"{sign}{number:6.2f}"
    if column in {"MRR", "nDCG"}:
        return f"{sign}{number:5.3f}"
    if column == "n":
        return f"{sign}{int(number):>4}"
    if delta:
        return f"{sign}{number:.1f}"
    return f"{number:.1f}"


def _difference(was: Any, now: Any) -> Any:
    """One value's change from the baseline, or None when either is absent.

    A percentage column is compared as a proportion, so a difference reads in the
    same unit the column does rather than as a fraction of a percentage.
    """

    if was is None or now is None:
        return None
    return float(now) - float(was)


def _verdict_line(record: dict[str, Any], verdict: str) -> str:
    source = (record.get("source_project") or {}).get("guard") or {}
    before = source.get("before") or {}
    if verdict == "verified":
        return (
            f"source project unchanged: {before.get('file_count', '?')} "
            f"files, {before.get('byte_count', '?')} bytes, digest "
            f"{str(before.get('digest', '?'))[:16]}"
        )
    if verdict == "source_project_changed":
        changed = ", ".join(str(item) for item in (source.get("differences") or [])[:8])
        more = len(source.get("differences") or []) - 8
        tail = f" and {more} more" if more > 0 else ""
        return (
            f"NOT A MEASUREMENT: the source project changed during the run: "
            f"{changed}{tail}"
        )
    return f"no arm of this run measured anything ({verdict})"


def _splits(record: dict[str, Any]) -> tuple[str, ...]:
    """Every split the run reported, in the order the arms name them."""

    names: list[str] = []
    for arm in record.get("arms") or []:
        for reports in arm.get("reports") or {}:
            if reports not in names:
                names.append(reports)
    if not names:
        for arm in record.get("arms") or []:
            names.append(_legacy_split(arm))
        names = [name for name in names if name]
    return tuple(names)


def _modes_of(arm: dict[str, Any], splits: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for split in splits:
        report = _read(_report_path(arm, split))
        for mode in (report or {}).get("summary") or {}:
            if mode not in found:
                found.append(str(mode))
    return found


def _report_path(arm: dict[str, Any], split: str | None) -> Any:
    """Where an arm's report for one split was written.

    An arm's `reports` map is keyed by split. A record written before splits
    existed carries one `report` and is read through it, because refusing a
    finished run's record would lose the numbers it already holds.
    """

    reports = arm.get("reports")
    if isinstance(reports, dict) and reports:
        if split and split in reports:
            return reports[split]
        return None
    if split and split not in {None, "all"}:
        return None
    return arm.get("report")


def _legacy_split(arm: dict[str, Any]) -> str:
    """The split name a record written before splits existed is read under."""

    from ..experiment.spec import SINGLE_SPLIT

    return SINGLE_SPLIT if arm.get("report") else ""


def _read(path: Any) -> dict[str, Any]:
    if not path:
        return {}
    location = Path(str(path))
    if not location.is_file():
        return {}
    try:
        document = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(
            f"An arm's report could not be read: {location}: {exc}. Delete the run "
            "and measure it again."
        ) from exc
    return document if isinstance(document, dict) else {}


def _quantile(runs: list[dict[str, Any]], key: str, fraction: float) -> float | None:
    """One quantile of a measured column, or None when no run carried it.

    Nearest-rank, so the value is one that was measured rather than an
    interpolation between two.
    """

    values = sorted(
        float(run[key]) for run in runs if isinstance(run.get(key), (int, float))
    )
    if not values:
        return None
    index = math.ceil(fraction * len(values)) - 1
    return values[min(max(index, 0), len(values) - 1)]


__all__ = [
    "ABSENT",
    "BUDGET_SETTINGS",
    "COLUMNS",
    "QUANTILES",
    "render_comparison",
    "rows_for_run",
]
