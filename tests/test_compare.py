"""That the table reads a report and computes no metric.

Every column is read out of a report the app's own harness wrote. These tests
hold that: a column the report stopped carrying prints as a dash rather than a
zero, the p95 is a value that was measured, and the differences are differences
from the first arm the specification listed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag_experiments.errors import ExperimentError
from rag_experiments.report.compare import (
    ABSENT,
    COLUMNS,
    render_comparison,
    rows_for_run,
)

MODE = "hybrid+rerank"


def _report(
    directory: Path, name: str, overall: dict[str, Any], seconds: list[float]
) -> Path:
    path = directory / name
    path.write_text(
        json.dumps(
            {
                "summary": {MODE: {"overall": overall, "per_class": {}}},
                "runs": [{"mode": MODE, "elapsed_seconds": value} for value in seconds],
            }
        ),
        encoding="utf-8",
    )
    return path


def _row(**overall: Any) -> dict[str, Any]:
    base = {
        "query_count": 30,
        "success_at_1": 0.5,
        "success_at_3": 0.6,
        "success_at_k": 0.7,
        "mrr": 0.6,
        "ndcg_at_k": 0.65,
        "document_success_at_k": 0.9,
        "mean_distinct_sources": 7.0,
        "mean_withheld": 1.5,
    }
    base.update(overall)
    return base


def _record(
    tmp_path: Path, arms: list[tuple[str, dict[str, Any], list[float]]]
) -> dict[str, Any]:
    entries = []
    for name, overall, seconds in arms:
        report = _report(tmp_path, f"{name}.json", overall, seconds)
        entries.append(
            {
                "arm": {"name": name, "kind": "settings"},
                "report": str(report),
            }
        )
    return {
        "verdict": "verified",
        "source_project": {
            "guard": {
                "unchanged": True,
                "differences": [],
                "before": {"file_count": 5, "byte_count": 100, "digest": "a" * 64},
            }
        },
        "arms": entries,
    }


def test_every_column_reads_a_key_a_row_carries() -> None:
    from rag_experiments.report.compare import _row as build_row

    keys = {key for _label, key, _doc in COLUMNS}
    row = build_row(
        {"arm": {"name": "a", "kind": "settings"}, "report": ""}, MODE, None
    )
    # A column whose key no row carries is a column that silently prints as a
    # dash, so the row is built from the column table and the two agree exactly.
    assert set(row) - {"present", "kind", "budgets"} == keys


def test_a_measured_arm_reads_every_column(tmp_path: Path) -> None:
    record = _record(tmp_path, [("baseline", _row(), [0.1, 0.2, 0.3, 0.4])])
    table = render_comparison(record)
    assert "baseline" in table
    assert "  30 " in table
    assert "50.0%" in table
    assert "0.600" in table
    assert "0.65" in table
    assert "7.0" in table
    assert "1.5" in table
    assert "0.40" in table, "the p95 is the slowest measured query"
    # A report written before the app recorded result-list contents has no such
    # columns, and a zero would read as an absence of duplication rather than of
    # measurement, so they print as dashes.
    for column in ("spans", "dup", "near", "1src", "rep%"):
        assert column in table
    assert "     -" in table


def test_a_metric_the_report_stopped_carrying_prints_as_a_dash(tmp_path: Path) -> None:
    partial = _row()
    del partial["mean_withheld"]
    record = _record(tmp_path, [("baseline", partial, [0.1])])
    rows = rows_for_run(record, MODE)
    assert rows[0]["mean_withheld"] is None
    table = render_comparison(record)
    assert ABSENT in table


def test_the_p95_is_a_measured_value_at_the_nearest_rank(tmp_path: Path) -> None:
    seconds = [float(value) for value in range(1, 21)]
    record = _record(tmp_path, [("baseline", _row(), seconds)])
    assert rows_for_run(record, MODE)[0]["p95_seconds"] == 19.0


def test_a_run_with_no_timing_carries_no_p95(tmp_path: Path) -> None:
    record = _record(tmp_path, [("baseline", _row(), [])])
    assert rows_for_run(record, MODE)[0]["p95_seconds"] is None


def test_differences_are_from_the_first_arm_the_specification_listed(
    tmp_path: Path,
) -> None:
    record = _record(
        tmp_path,
        [
            ("baseline", _row(success_at_1=0.5), [0.1]),
            ("trial", _row(success_at_1=0.7), [0.5]),
        ],
    )
    table = render_comparison(record)
    assert "differences from baseline, the first arm the specification listed" in table
    # 0.7 - 0.5, in the column's own unit.
    assert "+20.0%" in table or "20.0%" in table


def test_one_table_per_mode_because_a_mode_is_a_different_question(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path, [("baseline", _row(), [0.1])])
    assert f"mode: {MODE}" in render_comparison(record)
    assert "No arm of this run reported any of: bm25." in render_comparison(
        record, ("bm25",)
    )


def test_a_verdict_that_is_not_verified_leads_with_its_reason(tmp_path: Path) -> None:
    record = _record(tmp_path, [("baseline", _row(), [0.1])])
    record["verdict"] = "source_project_changed"
    record["source_project"]["guard"] = {
        "unchanged": False,
        "differences": [f"changed: sources/{index}.pdf" for index in range(10)],
        "before": {"file_count": 5, "byte_count": 100, "digest": "a" * 64},
    }
    table = render_comparison(record)
    assert "NOT A MEASUREMENT" in table
    assert "and 2 more" in table, "the count is stated, not just the first few"


def test_a_run_with_no_measured_mode_says_so() -> None:
    assert "No arm of this run reported" in render_comparison(
        {"verdict": "no_arm_measured", "arms": []}
    )


def test_an_unreadable_report_is_a_condition_not_a_dash(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("not json", encoding="utf-8")
    record = {
        "verdict": "verified",
        "source_project": {"guard": {"unchanged": True, "differences": []}},
        "arms": [{"arm": {"name": "a", "kind": "settings"}, "report": str(broken)}],
    }
    with pytest.raises(ExperimentError) as caught:
        render_comparison(record)
    assert "could not be read" in str(caught.value)


def test_an_absent_report_prints_as_dashes_rather_than_failing(tmp_path: Path) -> None:
    # A run whose arm directory was cleaned up has a record that cannot be read
    # into numbers. Saying so per column is honest; failing is not better.
    record = {
        "verdict": "verified",
        "source_project": {"guard": {"unchanged": True, "differences": []}},
        "arms": [
            {
                "arm": {"name": "a", "kind": "settings"},
                "report": str(tmp_path / "absent.json"),
            }
        ],
    }
    table = render_comparison(record)
    assert "No arm of this run reported" in table


def _two_split_record(tmp_path: Path) -> dict[str, Any]:
    """A record with two splits and arms at unequal budgets, as a sweep produces."""

    def arm(
        name: str, candidates: int, rerank: int, overall: dict[str, Any]
    ) -> dict[str, Any]:
        development = _report(tmp_path, f"{name}-d.json", overall, [0.1, 0.2, 0.3, 0.4])
        held_out = _report(
            tmp_path, f"{name}-h.json", {**overall, "query_count": 10}, [0.5, 0.6]
        )
        return {
            "arm": {"name": name, "kind": "settings"},
            "settings": {
                "values": {
                    "retrieval.maximum_candidates": candidates,
                    "retrieval.rerank_max_candidates": rerank,
                }
            },
            "reports": {"development": str(development), "held-out": str(held_out)},
        }

    return {
        "verdict": "verified",
        "source_project": {
            "guard": {
                "unchanged": True,
                "differences": [],
                "before": {"file_count": 5, "byte_count": 100, "digest": "d" * 64},
            }
        },
        "arms": [
            arm("baseline", 200, 50, _row()),
            arm("wider", 320, 50, _row(success_at_1=0.7)),
        ],
    }


def test_every_split_is_tabulated_because_a_held_out_number_stands_alone(
    tmp_path: Path,
) -> None:
    table = render_comparison(_two_split_record(tmp_path))
    assert "split: development" in table
    assert "split: held-out" in table
    assert table.count("split:") == 2


def test_one_split_can_be_tabulated_on_its_own(tmp_path: Path) -> None:
    table = render_comparison(_two_split_record(tmp_path), split="held-out")
    assert "split: held-out" in table
    assert "split: development" not in table


def test_the_splits_are_read_from_each_arms_own_reports(tmp_path: Path) -> None:
    record = _two_split_record(tmp_path)
    held = rows_for_run(record, MODE, "held-out")
    assert [row["n"] for row in held] == [10, 10]
    development = rows_for_run(record, MODE, "development")
    assert [row["n"] for row in development] == [30, 30]


def test_each_arms_budget_is_stated_above_the_table(tmp_path: Path) -> None:
    table = render_comparison(_two_split_record(tmp_path))
    assert "budgets (cand, rrank)" in table
    assert "baseline: cand=200 rrank=50" in table
    assert "wider: cand=320 rrank=50" in table


def test_arms_at_unequal_budgets_state_the_confound(tmp_path: Path) -> None:
    # A candidate-window sweep varies the window on purpose and a policy ablation
    # is only comparable at a fixed one, so the table states that a difference is
    # also a difference in budget rather than calling the run invalid.
    table = render_comparison(_two_split_record(tmp_path))
    assert "differs across these arms" in table
    assert "difference in budget as well as in policy" in table


def test_arms_at_equal_budgets_carry_no_warning(tmp_path: Path) -> None:
    record = _two_split_record(tmp_path)
    for arm in record["arms"]:
        arm["settings"]["values"]["retrieval.maximum_candidates"] = 200
    table = render_comparison(record)
    assert "differs across these arms" not in table
    assert "differences from baseline" in table


def test_a_budget_the_record_does_not_state_prints_as_a_dash(
    tmp_path: Path,
) -> None:
    record = _two_split_record(tmp_path)
    for arm in record["arms"]:
        arm["settings"] = {"values": {}}
    table = render_comparison(record)
    assert "cand=- rrank=-" in table


def test_both_latency_quantiles_are_reported(tmp_path: Path) -> None:
    seconds = [float(value) for value in range(1, 21)]
    record = _record(tmp_path, [("baseline", _row(), seconds)])
    row = rows_for_run(record, MODE)[0]
    assert row["p50_seconds"] == 10.0
    assert row["p95_seconds"] == 19.0
    table = render_comparison(record)
    assert "p50 s" in table and "p95 s" in table


def test_a_record_written_before_splits_are_read_under_one_name(tmp_path: Path) -> None:
    # A finished run's record carries one report and must stay readable rather
    # than being refused because the shape it was written in has since changed.
    record = {
        "verdict": "verified",
        "source_project": {"guard": {"unchanged": True, "differences": []}},
        "arms": [
            {
                "arm": {"name": "baseline", "kind": "settings"},
                "report": str(_report(tmp_path, "legacy.json", _row(), [0.1, 0.2])),
            }
        ],
    }
    assert "split: all" in render_comparison(record)


def test_the_verdict_leads_a_run_with_no_readable_number(tmp_path: Path) -> None:
    record = _record(tmp_path, [("baseline", _row(), [])])
    del record["arms"][0]["report"]
    record["arms"][0]["reports"] = {}
    out = render_comparison(record)
    assert out.startswith("source project unchanged")
    assert "No arm of this run reported" in out


def _redundancy_record(tmp_path: Path) -> dict[str, Any]:
    """A record from a report that carries result-list contents."""

    def arm(name: str, exact: float, spans: float, repeated: float) -> dict[str, Any]:
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "summary": {
                        MODE: {
                            "overall": {
                                **_row(),
                                "mean_distinct_evidence_spans": spans,
                                "mean_exact_duplicate_slots": exact,
                                "mean_near_duplicate_slots": 0.0,
                                "mean_same_source_pairs": 2.0,
                            },
                            "repeated_slot_rate": repeated,
                        }
                    },
                    "runs": [{"mode": MODE, "elapsed_seconds": 0.1}],
                }
            ),
            encoding="utf-8",
        )
        return {"arm": {"name": name, "kind": "settings"}, "report": str(path)}

    return {
        "verdict": "verified",
        "source_project": {
            "guard": {"unchanged": True, "differences": [], "before": {}}
        },
        "arms": [arm("baseline", 0.0, 10.0, 0.0), arm("deduplicated", 2.0, 8.0, 0.0)],
    }


def test_the_result_list_columns_read_the_reports_own_values(tmp_path: Path) -> None:
    rows = rows_for_run(_redundancy_record(tmp_path), MODE)
    assert rows[0]["mean_distinct_evidence_spans"] == 10.0
    assert rows[0]["mean_exact_duplicate_slots"] == 0.0
    assert rows[0]["mean_same_source_pairs"] == 2.0
    assert rows[1]["mean_distinct_evidence_spans"] == 8.0
    assert rows[1]["mean_exact_duplicate_slots"] == 2.0


def test_the_repeated_slot_rate_is_read_from_the_modes_own_summary(
    tmp_path: Path,
) -> None:
    record = _redundancy_record(tmp_path)
    path = tmp_path / "repeated.json"
    path.write_text(
        json.dumps(
            {
                "summary": {
                    MODE: {"overall": _row(), "repeated_slot_rate": 0.11875},
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    record["arms"] = [{"arm": {"name": "a", "kind": "settings"}, "report": str(path)}]
    assert rows_for_run(record, MODE)[0]["repeated_slot_rate"] == 0.11875
    # It is a percentage column, so a reader sees 11.9 rather than 0.119.
    assert "11.9%" in render_comparison(record)


def test_a_change_the_ranking_metrics_cannot_see_still_prints(tmp_path: Path) -> None:
    # The second arm loses three spans and gains two duplicates while every
    # quality column is identical. That difference has to be readable, because it
    # is the one a known-item score is blind to.
    record = _redundancy_record(tmp_path)
    table = render_comparison(record)
    assert "spans" in table and "dup" in table
    assert "-2.0" in table, "the span loss is a difference from the baseline"
    assert "+2.0" in table, "the duplicate gain is a difference from the baseline"
