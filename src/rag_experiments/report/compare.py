"""Read a run's arms side by side, without computing a number.

Every column is read out of a report the app's own harness wrote. The comparison
adds nothing: it does not average, weight, rank, or decide. A column this module
cannot find is printed as a dash, because a metric the app's harness stopped
reporting is a fact about the harness and printing a zero for it would be a
different one.

A run's first arm is the row every other arm is read against, so a difference is
a difference from the arm the specification listed first rather than from a
number chosen after the fact.
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
    ("n", "n", "Judged queries the arm actually ran."),
    ("succ@1", "success_at_1", "The target passage ranked first."),
    ("succ@3", "success_at_3", "The target passage in the first three."),
    ("succ@k", "success_at_k", "The target passage inside the depth shown."),
    ("MRR", "mrr", "Mean reciprocal rank of the target passage."),
    ("nDCG", "ndcg_at_k", "Normalized discounted cumulative gain at that depth."),
    ("doc@k", "document_success_at_k", "The target's document retrieved at all."),
    ("srcs", "mean_distinct_sources", "Distinct sources those passages came from."),
    ("withheld", "mean_withheld", "Candidates a relevance gate rejected."),
    (
        "p95 s",
        "p95_seconds",
        "The slowest measured query, in seconds, at the nearest rank.",
    ),
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
    "srcs": 5,
    "withheld": 9,
    "p95 s": 7,
}


def rows_for_run(record: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    """One row per arm of a run, for one measured mode.

    A mode the record has no row for is an empty table rather than an error: a
    specification that asked for one mode out of four is a run that measured one
    mode, and the record says which.
    """

    rows: list[dict[str, Any]] = []
    for arm in record.get("arms") or []:
        rows.append(_row(arm, mode))
    return rows


def _row(arm: dict[str, Any], mode: str) -> dict[str, Any]:
    """One arm's numbers for one mode, keyed as `COLUMNS` reads them.

    A key here and a key in `COLUMNS` that drift apart is a column that silently
    prints as a dash, so the row is built with the column table's own names.
    """

    report = _read(arm.get("report"))
    summary = (report or {}).get("summary") or {}
    measured = summary.get(mode) or {}
    overall = measured.get("overall") or {}
    per_query = [
        run for run in (report or {}).get("runs") or [] if str(run.get("mode")) == mode
    ]
    return {
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
        "p95_seconds": _percentile(per_query, "elapsed_seconds"),
        "present": bool(measured),
    }


def render_comparison(record: dict[str, Any], modes: tuple[str, ...] = ()) -> str:
    """The table a reader compares arms with, as text.

    One block per mode, because a mode is a different question asked the same
    queries, and mixing them into one row would hide which question a number
    answered.
    """
    wanted = modes or tuple(
        dict.fromkeys(
            str(mode) for arm in record.get("arms") or [] for mode in _modes_of(arm)
        )
    )
    # The verdict leads whatever else is printed, including when there is nothing
    # else to print: a run with no readable number is exactly when a reader needs
    # to be told whether the corpus it measured is the corpus on disk.
    header = _verdict_line(record, str(record.get("verdict") or ""))
    if not wanted:
        return f"{header}\n\nNo arm of this run reported a measured mode."

    blocks: list[str] = []
    for mode in wanted:
        rows = rows_for_run(record, mode)
        if not any(row["present"] for row in rows):
            continue
        blocks.append(_table(mode, rows))
    if not blocks:
        return (
            f"{header}\n\n"
            f"No arm of this run reported any of: {', '.join(wanted)}.\n"
            "The specification asked for them under `harness.modes`, and the "
            "reports each arm wrote do not carry them."
        )
    return "\n".join([header, ""] + ["\n\n".join(blocks)])


def _table(mode: str, rows: list[dict[str, Any]]) -> str:
    names = [row["arm"] for row in rows]
    arm_width = max([len(name) for name in names] + [3])
    lines = [f"mode: {mode}"]
    lines.append(_header(arm_width))
    lines.append("-" * len(lines[-1]))
    for row in rows:
        lines.append(_line(row, arm_width))
    baseline = rows[0]
    if len(rows) > 1:
        lines.append("")
        lines.append(
            f"differences from {baseline['arm']}, the first arm the specification "
            f"listed:"
        )
        for row in rows[1:]:
            lines.append(_delta(baseline, row, arm_width))
    return "\n".join(lines)


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


def _delta(baseline: dict[str, Any], row: dict[str, Any], arm_width: int) -> str:
    cells = [f"{row['arm']:<{arm_width}}"]
    for name, key, _doc in COLUMNS[1:]:
        cells.append(
            f"{_format(_difference(baseline.get(key), row.get(key)), name):>{WIDTHS[name]}}"
        )
    return "".join(cells)


def _format(value: Any, column: str) -> str:
    if value is None:
        return ABSENT
    if column in PERCENT_COLUMNS:
        return f"{100.0 * float(value):6.1f}%"
    if column == "p95 s":
        return f"{float(value):6.2f}"
    if column in {"MRR", "nDCG"}:
        return f"{float(value):5.3f}"
    if column == "n":
        return f"{int(value):>4}"
    return f"{float(value):.1f}"


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
    if verdict == "verified":
        return (
            f"source project unchanged: {source.get('before', {}).get('file_count', '?')} "
            f"files, {source.get('before', {}).get('byte_count', '?')} bytes, digest "
            f"{str(source.get('before', {}).get('digest', '?'))[:16]}"
        )
    if verdict == "source_project_changed":
        changed = ", ".join(str(item) for item in (source.get("differences") or [])[:8])
        more = len(source.get("differences") or []) - 8
        tail = f" and {more} more" if more > 0 else ""
        return f"NOT A MEASUREMENT: the source project changed during the run: {changed}{tail}"
    return f"no arm of this run measured anything ({verdict})"


def _modes_of(arm: dict[str, Any]) -> list[str]:
    report = _read(arm.get("report"))
    return list((report or {}).get("summary") or {})


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


def _percentile(runs: list[dict[str, Any]], key: str) -> float | None:
    """The p95 of a measured column, or None when no run carried it.

    The slowest queries are what a reader feels, and a mean hides them, so the
    column is the p95 rather than a second average. The index is the nearest rank,
    so the value returned is one that was actually measured rather than an
    interpolation between two.
    """

    values = sorted(
        float(run[key]) for run in runs if isinstance(run.get(key), (int, float))
    )
    if not values:
        return None
    index = math.ceil(0.95 * len(values)) - 1
    return values[min(max(index, 0), len(values) - 1)]


__all__ = ["ABSENT", "COLUMNS", "render_comparison", "rows_for_run"]
