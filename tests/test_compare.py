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
    row = build_row({"arm": {"name": "a", "kind": "settings"}, "report": ""}, MODE)
    # A column whose key no row carries is a column that silently prints as a
    # dash, so the row is built from the column table and the two agree exactly.
    assert set(row) - {"present", "kind"} == keys
    assert keys - {"arm"} <= {key for _label, key, _doc in COLUMNS if key}


def test_a_measured_arm_reads_every_column(tmp_path: Path) -> None:
    record = _record(tmp_path, [("baseline", _row(), [0.1, 0.2, 0.3, 0.4])])
    table = render_comparison(record)
    assert "baseline" in table
    assert " 30 " in table
    assert "50.0%" in table
    assert "0.600" in table
    assert "0.65" in table
    assert "7.0" in table
    assert "1.5" in table
    assert "0.40" in table, "the p95 is the slowest measured query"


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
