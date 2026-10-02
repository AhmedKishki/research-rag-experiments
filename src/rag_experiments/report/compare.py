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

import hashlib
import json
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
    (
        "texts",
        "mean_distinct_normalized_texts",
        "Distinct ordered normalized texts, not claims.",
    ),
    (
        "dup",
        "mean_exact_duplicate_slots",
        "Slots repeating another returned passage verbatim.",
    ),
    (
        "lexov",
        "mean_lexical_containment_slots",
        "Lexically contained slots; not a semantic redundancy judgment.",
    ),
    (
        "1src",
        "mean_same_source_pairs",
        "Slots sharing a source file with another slot.",
    ),
    ("srcs", "mean_distinct_sources", "Distinct sources those passages came from."),
    (
        "xfam%",
        "repeated_slot_rate_cross_target_family",
        "Repeated slots across different target families; not proof of hubness.",
    ),
    ("withheld", "mean_withheld", "Passages withheld from the answer."),
    (
        "rej",
        "mean_dense_rejected_below_floor",
        "Dense candidates the cosine gate rejected.",
    ),
    ("p50 s", "p50_seconds", "The median measured query, in seconds."),
    ("p95 s", "p95_seconds", "The reported 95th percentile, in seconds."),
)

#: The settings that decide how much work a query was allowed, and the label each
#: is printed under. Read from the arm's pinned settings, because that is the
#: authority for what the engine was told; a budget the table cannot state is a
#: comparison a reader cannot judge.
BUDGET_SETTINGS: tuple[tuple[str, str], ...] = (
    ("retrieval.minimum_candidates", "min-cand"),
    ("retrieval.maximum_candidates", "cand"),
    ("retrieval.rerank_max_candidates", "rrank"),
    ("retrieval.rerank_window_multiple", "multiple"),
    ("retrieval.rerank_window_floor", "floor"),
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
    "texts": 6,
    "dup": 5,
    "lexov": 6,
    "1src": 5,
    "srcs": 5,
    "xfam%": 7,
    "withheld": 9,
    "rej": 6,
    "p50 s": 7,
    "p95 s": 7,
}


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

    report = _read_arm_report(arm, split)
    summary = report.get("summary") or {}
    measured = summary.get(mode) or {}
    overall = measured.get("overall") or {}
    per_query = [run for run in report.get("runs", []) if run.get("mode") == mode]
    values = (arm.get("settings") or {}).get("values") or {}
    corrected = _report_schema(report) >= 3 if report else False
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
        # The gate's own counts, which withheld_candidates never carries: that
        # count is about the answer, this one about the gate that ran before
        # fusion, and a search can report zero of the first while rejecting a
        # hundred of the second.
        "mean_dense_rejected_below_floor": overall.get(
            "mean_dense_rejected_below_floor"
        ),
        "mean_dense_admitted_below_floor": overall.get(
            "mean_dense_admitted_below_floor"
        ),
        "mean_distinct_normalized_texts": overall.get("mean_distinct_normalized_texts")
        if corrected
        else None,
        "mean_exact_duplicate_slots": (
            overall.get("mean_exact_duplicate_slots") if corrected else None
        ),
        "mean_lexical_containment_slots": overall.get("mean_lexical_containment_slots")
        if corrected
        else None,
        "mean_same_source_pairs": overall.get("mean_same_source_pairs"),
        # Measured across the mode's queries rather than within one query, so it
        # is reported per mode beside the per-query means rather than as one.
        "repeated_slot_rate_cross_target_family": measured.get(
            "repeated_slot_rate_cross_target_family"
        )
        if corrected
        else None,
        "p50_seconds": overall.get("p50_seconds"),
        "p95_seconds": overall.get("p95_seconds"),
        "budgets": {key: values.get(key) for key, _label in BUDGET_SETTINGS},
        "actual_depths": sorted(
            {
                r["candidate_depth"]
                for r in per_query
                if isinstance(r.get("candidate_depth"), int)
            }
        ),
        "actual_windows": sorted(
            {
                r["rerank_window"]
                for r in per_query
                if isinstance(r.get("rerank_window"), int)
            }
        ),
        "work_by_query": {
            str(r["query_id"]): (r["candidate_depth"], r["rerank_window"])
            for r in per_query
            if r.get("query_id")
            and isinstance(r.get("candidate_depth"), int)
            and isinstance(r.get("rerank_window"), int)
        },
        "rerank_confirmed": all(
            r.get("reranked") is True and not r.get("rerank_fallback")
            for r in per_query
        )
        if any(r.get("rerank_requested") is True for r in per_query)
        or ("rerank" in mode and per_query)
        else None,
        "report_schema": report.get("schema_version"),
        "present": mode in summary,
        "degraded": bool(report.get("degraded"))
        or any(r.get("rerank_fallback") for r in per_query),
        "incomplete_text": sum(
            (r.get("duplicate_measure_coverage") or {}).get("slots_missing_text", 0)
            for r in per_query
        ),
        "pool_counts": sorted(
            {
                r["candidate_count"]
                for r in per_query
                if isinstance(r.get("candidate_count"), int)
            }
        ),
        "collapse_counts": sorted(
            {
                r["collapsed_count"]
                for r in per_query
                if isinstance(r.get("collapsed_count"), int)
            }
        ),
        "stage_diagnostics": {
            key: value
            for key, value in overall.items()
            if key.startswith(
                ("mean_target_present_", "target_stage_evaluated_queries_")
            )
        },
        "no_answer_support": report.get("no_answer_support"),
        "latency_protocol": report.get(
            "latency_protocol", "single-pass; repeat/order effects are not estimated"
        ),
    }
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
    schemas = {
        _report_schema(r)
        for arm in record.get("arms", [])
        for one in splits
        if (r := _read_arm_report(arm, one))
    }
    if len(schemas) > 1:
        raise ExperimentError(
            "Reports use different metric schemas; do not compare their deltas"
        )
    definitions = {
        json.dumps(
            {
                "metrics": r.get("metric_definitions"),
                "lexical": r.get("lexical_containment"),
                "scorer": (r.get("harness") or {}).get("script_sha256"),
            },
            sort_keys=True,
        )
        for arm in record.get("arms", [])
        for one in splits
        if (r := _read_arm_report(arm, one))
    }
    if len(definitions) > 1:
        raise ExperimentError(
            "Reports use different metric definitions; deltas are not comparable"
        )
    if schemas and min(schemas) < 3:
        header += (
            "\nLEGACY REPORT: exact/near word-set and evidence-span metrics are "
            "superseded; they are not displayed as corrected metrics."
        )
    if any(
        _report_path(arm, one) and not _expected_hash(arm, one)
        for arm in record.get("arms", [])
        for one in splits
    ):
        header += "\nUNVERIFIED ARTIFACTS: this record lacks report hashes; displayed bytes cannot be authenticated."
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
    lines.append(_observed_budgets(rows))
    if any(r["no_answer_support"] for r in rows):
        lines.append(
            "no-answer scope: "
            + "   ".join(
                f"{r['arm']}: {json.dumps(r['no_answer_support'], sort_keys=True)}"
                for r in rows
            )
        )
    if any(r["degraded"] for r in rows):
        lines.append(
            "DEGRADED: reranking fallback or report degradation; do not select a policy from this block."
        )
    if any(r["incomplete_text"] for r in rows):
        lines.append(
            "INCOMPLETE TEXT COVERAGE: lexical diagnostics are unavailable for missing passages."
        )
    lines.append(_header(arm_width))
    lines.append("-" * len(lines[-1]))
    for row in rows:
        lines.append(_line(row, arm_width))
    if any(r["stage_diagnostics"] for r in rows):
        lines.append("designated-target stage presence (known queries / all queries):")
        for row in rows:
            fields = row["stage_diagnostics"]
            stages = [
                key.removeprefix("mean_target_present_")
                for key in fields
                if key.startswith("mean_target_present_")
            ]
            lines.append(
                f"  {row['arm']}: "
                + "  ".join(
                    f"{stage}={ABSENT if fields['mean_target_present_' + stage] is None else format(fields['mean_target_present_' + stage], '.3f')} "
                    f"({fields.get('target_stage_evaluated_queries_' + stage, ABSENT)}/{row['n']})"
                    for stage in stages
                )
            )
    baseline = rows[0]
    if len(rows) > 1 and (not baseline["present"] or not baseline.get("n")):
        lines.append("NO DELTAS: the designated baseline has no measured queries.")
    elif len(rows) > 1 and not any(r["degraded"] for r in rows):
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
                "  requested or observed budgets differ or are incomplete across these arms; "
                "a policy-only effect cannot be inferred. A policy ablation "
                "holds observed work fixed."
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
        if any(budgets.get(key) is None for key, _ in BUDGET_SETTINGS):
            return False
        seen.add(tuple(budgets.get(key) for key, _label in BUDGET_SETTINGS))
    if any(not r["actual_depths"] or not r["actual_windows"] for r in rows):
        return False
    if any(not r["work_by_query"] or r["rerank_confirmed"] is False for r in rows):
        return False
    if any(r["work_by_query"] != rows[0]["work_by_query"] for r in rows[1:]):
        return False
    return (
        len(seen) <= 1
        and len({(tuple(r["actual_depths"]), tuple(r["actual_windows"])) for r in rows})
        <= 1
    )


def _header(arm_width: int) -> str:
    cells = [f"{'arm':<{arm_width}}"]
    for name, _key, _doc in COLUMNS[1:]:
        cells.append(f"{name:>{WIDTHS[name]}}")
    return " ".join(cells)


def _line(row: dict[str, Any], arm_width: int) -> str:
    cells = [f"{row['arm']:<{arm_width}}"]
    for name, key, _doc in COLUMNS[1:]:
        cells.append(f"{_format(row.get(key), name):>{WIDTHS[name]}}")
    return " ".join(cells)


def _delta_line(baseline: dict[str, Any], row: dict[str, Any], arm_width: int) -> str:
    cells = [f"{row['arm']:<{arm_width}}"]
    for name, key, _doc in COLUMNS[1:]:
        difference = _difference(baseline.get(key), row.get(key))
        cells.append(f"{_format(difference, name, delta=True):>{WIDTHS[name]}}")
    return " ".join(cells)


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
    scale = 100.0 if column in PERCENT_COLUMNS or column == "xfam%" else 1.0
    digits = (
        2 if column in {"p50 s", "p95 s"} else 3 if column in {"MRR", "nDCG"} else 1
    )
    rounded = round(number * scale, digits)
    if rounded == 0:
        rounded = 0.0
    sign = "+" if delta and rounded > 0 else ""
    if scale == 100.0:
        return f"{sign}{rounded:.1f}%"
    if column in {"p50 s", "p95 s", "MRR", "nDCG"}:
        return f"{sign}{rounded:.{digits}f}"
    if column == "n":
        return f"{sign}{int(number):>4}"
    return f"{sign}{rounded:.1f}"


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
    if verdict == "verified" and source.get("state") == "unknown":
        return "NOT A VERIFIED MEASUREMENT: source integrity could not be checked"
    if verdict == "verified":
        if source.get("unchanged") is not True or not before.get("digest"):
            return "NOT A VERIFIED MEASUREMENT: the recorded source guard is incomplete or inconsistent"
        return (
            f"source project unchanged: {before.get('file_count', '?')} "
            f"files, {before.get('byte_count', '?')} bytes, digest "
            f"{str(before.get('digest', '?'))[:16]}"
            + (f"; guard scope: {source['note']}" if source.get("note") else "")
        )
    if verdict == "source_project_changed":
        changed = ", ".join(str(item) for item in (source.get("differences") or [])[:8])
        more = len(source.get("differences") or []) - 8
        tail = f" and {more} more" if more > 0 else ""
        return (
            f"NOT A MEASUREMENT: the source project changed during the run: "
            f"{changed}{tail}"
        )
    if verdict in {"failed", "incomplete", "engine_source_changed", "interrupted"}:
        state = source.get("state", "unknown")
        return f"NOT A COMPLETE MEASUREMENT: {verdict}; source guard {state}"
    if source.get("state") == "unknown":
        return "NOT A VERIFIED MEASUREMENT: source integrity could not be checked"
    if verdict == "no_arm_measured":
        return f"no arm of this run measured anything ({verdict})"
    return (
        f"UNKNOWN RUN VERDICT: {verdict or 'absent'}; verification is not established"
    )


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
        report = _read_arm_report(arm, split)
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


def _read_arm_report(arm: dict[str, Any], split: str | None) -> dict[str, Any]:
    """Check recorded report bytes before presenting their numbers."""

    path = _report_path(arm, split)
    expected = _expected_hash(arm, split)
    for measurement in arm.get("measure") or []:
        if isinstance(measurement, dict) and measurement.get("split") == split:
            if measurement.get("report") != path:
                raise ExperimentError("Recorded measurement/report paths disagree")
            expected = (
                expected
                or measurement.get("report_sha256")
                or (measurement.get("report_facts") or {}).get("sha256")
            )
    if path and expected:
        location = Path(path)
        try:
            actual = hashlib.sha256(location.read_bytes()).hexdigest()
        except OSError as exc:
            raise ExperimentError(
                f"Report integrity check failed: {location}: {exc}"
            ) from exc
        if actual != expected:
            raise ExperimentError(f"Report integrity check failed: {location}")
    return _read(path)


def _expected_hash(arm: dict[str, Any], split: str | None) -> str | None:
    expected = (arm.get("report_hashes") or {}).get(split)
    for measurement in arm.get("measure") or []:
        if isinstance(measurement, dict) and measurement.get("split") == split:
            return (
                expected
                or measurement.get("report_sha256")
                or (measurement.get("report_facts") or {}).get("sha256")
            )
    return expected


def _report_schema(report: dict[str, Any]) -> int:
    value = report.get("schema_version", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 2, 3}:
        raise ExperimentError(f"Unsupported report schema: {value!r}")
    return value


def _observed_budgets(rows: list[dict[str, Any]]) -> str:
    """Distinguish requested caps from observed branch depths and scored windows."""

    return "observed: " + "   ".join(
        f"{r['arm']}: depth={r['actual_depths'] or ABSENT} "
        f"rerank_window={r['actual_windows'] or ABSENT} "
        f"rerank_applied={r['rerank_confirmed'] if r['rerank_confirmed'] is not None else ABSENT} "
        f"post_collapse_pool={r['pool_counts'] or ABSENT} "
        f"collapsed={r['collapse_counts'] or ABSENT}"
        for r in rows
    )


__all__ = [
    "ABSENT",
    "BUDGET_SETTINGS",
    "COLUMNS",
    "render_comparison",
    "rows_for_run",
]
