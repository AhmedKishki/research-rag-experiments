"""That the table reads a report and computes no metric.

Every column is read out of a report the app's own harness wrote. These tests
hold that: a column the report stopped carrying prints as a dash rather than a
zero, the p95 is a value that was measured, and the differences are differences
from the first arm the specification listed.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    ordered = sorted(seconds)
    timing = {
        name: ordered[math.ceil(q * len(ordered)) - 1] if ordered else None
        for name, q in (("p50_seconds", 0.5), ("p95_seconds", 0.95))
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "summary": {MODE: {"overall": {**overall, **timing}, "per_class": {}}},
                "runs": [
                    {
                        "mode": MODE,
                        "query_id": f"q{position}",
                        "elapsed_seconds": value,
                        "candidate_depth": 40,
                        "rerank_window": 20,
                        "reranked": True,
                    }
                    for position, value in enumerate(seconds)
                ],
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
    # dash, so the row carries exactly the printed keys plus named extras. The
    # rescued count is read and carried because it is the gate's other half, but
    # only the rejected count is printed: a column of two adjacent counts would
    # put one number in the reach of a mistake.
    extras = {
        "present",
        "kind",
        "budgets",
        "mean_dense_admitted_below_floor",
        "actual_depths",
        "actual_windows",
        "rerank_confirmed",
        "report_schema",
        "work_by_query",
        "degraded",
        "incomplete_text",
        "pool_counts",
        "collapse_counts",
        "stage_diagnostics",
        "no_answer_support",
        "latency_protocol",
    }
    assert set(row) - extras == keys


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
    for column in ("texts", "dup", "lexov", "1src", "xfam%"):
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
                    "retrieval.minimum_candidates": 40,
                    "retrieval.maximum_candidates": candidates,
                    "retrieval.rerank_max_candidates": rerank,
                    "retrieval.rerank_window_multiple": 2,
                    "retrieval.rerank_window_floor": 10,
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
    assert "budgets (min-cand, cand, rrank, multiple, floor)" in table
    assert "baseline: min-cand=40 cand=200 rrank=50" in table
    assert "wider: min-cand=40 cand=320 rrank=50" in table


def test_arms_at_unequal_budgets_state_the_confound(tmp_path: Path) -> None:
    # A candidate-window sweep varies the window on purpose and a policy ablation
    # is only comparable at a fixed one, so the table states that a difference is
    # also a difference in budget rather than calling the run invalid.
    table = render_comparison(_two_split_record(tmp_path))
    assert "budgets differ or are incomplete" in table
    assert "policy-only effect cannot be inferred" in table


def test_arms_at_equal_budgets_carry_no_warning(tmp_path: Path) -> None:
    record = _two_split_record(tmp_path)
    for arm in record["arms"]:
        arm["settings"]["values"]["retrieval.maximum_candidates"] = 200
    table = render_comparison(record)
    assert "budgets differ or are incomplete" not in table
    assert "differences from baseline" in table


def test_a_budget_the_record_does_not_state_prints_as_a_dash(
    tmp_path: Path,
) -> None:
    record = _two_split_record(tmp_path)
    for arm in record["arms"]:
        arm["settings"] = {"values": {}}
    table = render_comparison(record)
    assert "cand=- rrank=-" in table
    assert "observed:" in table


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
                    "schema_version": 3,
                    "summary": {
                        MODE: {
                            "overall": {
                                **_row(),
                                "mean_distinct_normalized_texts": spans,
                                "mean_exact_duplicate_slots": exact,
                                "mean_lexical_containment_slots": 0.0,
                                "mean_same_source_pairs": 2.0,
                            },
                            "repeated_slot_rate_cross_target_family": repeated,
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
    assert rows[0]["mean_distinct_normalized_texts"] == 10.0
    assert rows[0]["mean_exact_duplicate_slots"] == 0.0
    assert rows[0]["mean_same_source_pairs"] == 2.0
    assert rows[1]["mean_distinct_normalized_texts"] == 8.0
    assert rows[1]["mean_exact_duplicate_slots"] == 2.0


def test_the_repeated_slot_rate_is_read_from_the_modes_own_summary(
    tmp_path: Path,
) -> None:
    record = _redundancy_record(tmp_path)
    path = tmp_path / "repeated.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "summary": {
                    MODE: {
                        "overall": _row(),
                        "repeated_slot_rate_cross_target_family": 0.11875,
                    },
                },
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    record["arms"] = [{"arm": {"name": "a", "kind": "settings"}, "report": str(path)}]
    assert (
        rows_for_run(record, MODE)[0]["repeated_slot_rate_cross_target_family"]
        == 0.11875
    )
    # It is a percentage column, so a reader sees 11.9 rather than 0.119.
    assert "11.9%" in render_comparison(record)


def test_a_change_the_ranking_metrics_cannot_see_still_prints(tmp_path: Path) -> None:
    # The second arm loses three spans and gains two duplicates while every
    # quality column is identical. That difference has to be readable, because it
    # is the one a known-item score is blind to.
    record = _redundancy_record(tmp_path)
    table = render_comparison(record)
    assert "texts" in table and "dup" in table
    assert "-2.0" in table, "the span loss is a difference from the baseline"
    assert "+2.0" in table, "the duplicate gain is a difference from the baseline"


def test_the_gate_columns_are_read_and_are_separate_from_withheld(
    tmp_path: Path,
) -> None:
    # withheld_candidates counts what left the answer; the gate counts what it
    # dropped before fusion. A run where the first is zero and the second is not is
    # ordinary, and printing only the first would report the gate as inert.
    path = tmp_path / "gated.json"
    path.write_text(
        json.dumps(
            {
                "summary": {
                    MODE: {
                        "overall": {
                            **_row(mean_withheld=0.0),
                            "mean_dense_rejected_below_floor": 13.3,
                            "mean_dense_admitted_below_floor": 12.9,
                        },
                        "repeated_slot_rate": 0.0,
                    }
                },
                "runs": [{"mode": MODE, "elapsed_seconds": 0.1}],
            }
        ),
        encoding="utf-8",
    )
    row = rows_for_run(
        {"arms": [{"arm": {"name": "a", "kind": "settings"}, "report": str(path)}]},
        MODE,
    )[0]
    assert row["mean_withheld"] == 0.0
    assert row["mean_dense_rejected_below_floor"] == 13.3
    assert row["mean_dense_admitted_below_floor"] == 12.9
    table = render_comparison(
        {
            "verdict": "verified",
            "source_project": {"guard": {"unchanged": True, "differences": []}},
            "arms": [{"arm": {"name": "a", "kind": "settings"}, "report": str(path)}],
        }
    )
    assert "rej" in table
    assert "13.3" in table


def test_observed_depths_are_not_substituted_with_configured_caps(tmp_path):
    record = _two_split_record(tmp_path)
    table = render_comparison(record)
    assert "cand=320" in table
    assert "wider: depth=[40] rerank_window=[20] rerank_applied=True" in table


def test_report_hash_mismatch_refuses_to_display_replaced_evidence(tmp_path):
    record = _two_split_record(tmp_path)
    arm = record["arms"][0]
    path = Path(arm["reports"]["development"])
    arm["report_hashes"] = {
        "development": hashlib.sha256(path.read_bytes()).hexdigest()
    }
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ExperimentError, match="integrity check failed"):
        render_comparison(record)


def test_mixing_metric_schemas_cannot_produce_comparable_deltas(tmp_path):
    record = _two_split_record(tmp_path)
    path = Path(record["arms"][0]["reports"]["development"])
    report = json.loads(path.read_text())
    report["schema_version"] = 2
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ExperimentError, match="different metric schemas"):
        render_comparison(record)


def test_legacy_metrics_are_not_relabelled_as_corrected_metrics(tmp_path):
    record = _redundancy_record(tmp_path)
    for arm in record["arms"]:
        path = Path(arm["report"])
        report = json.loads(path.read_text())
        report["schema_version"] = 2
        for payload in report["summary"].values():
            payload["overall"]["mean_distinct_evidence_spans"] = 10
            payload["overall"].pop("mean_distinct_normalized_texts", None)
            payload["overall"]["mean_near_duplicate_slots"] = 3
            payload["overall"].pop("mean_lexical_containment_slots", None)
            payload["repeated_slot_rate"] = 0.3
            payload.pop("repeated_slot_rate_cross_target_family", None)
        path.write_text(json.dumps(report), encoding="utf-8")
    row = rows_for_run(record, MODE)[0]
    assert row["mean_distinct_normalized_texts"] is None
    assert row["mean_exact_duplicate_slots"] is None
    assert row["mean_lexical_containment_slots"] is None
    assert row["repeated_slot_rate_cross_target_family"] is None
    assert "LEGACY REPORT" in render_comparison(record)


@pytest.mark.parametrize("verdict", ["failed", "incomplete", "engine_source_changed"])
def test_failed_run_never_prints_a_successful_verification(verdict):
    record = {
        "verdict": verdict,
        "source_project": {"guard": {"state": "unknown"}},
        "arms": [],
    }
    out = render_comparison(record)
    assert "NOT A COMPLETE MEASUREMENT" in out
    assert "source guard unknown" in out
    assert "source project unchanged" not in out


def test_latency_is_read_from_report_not_recomputed_by_toolkit(tmp_path):
    record = _record(tmp_path, [("baseline", _row(), [1, 2, 3])])
    path = Path(record["arms"][0]["report"])
    report = json.loads(path.read_text())
    report["summary"][MODE]["overall"]["p95_seconds"] = 42
    path.write_text(json.dumps(report), encoding="utf-8")
    assert rows_for_run(record, MODE)[0]["p95_seconds"] == 42


def test_rounded_negative_zero_does_not_look_like_a_difference():
    from rag_experiments.report.compare import _format

    assert _format(-0.0001, "p50 s", delta=True) == "0.00"
    assert _format(-0.0001, "MRR", delta=True) == "0.000"
    assert _format(-0.000001, "succ@k", delta=True) == "0.0%"


def test_equal_window_ranges_do_not_hide_different_work_per_query(tmp_path):
    record = _two_split_record(tmp_path)
    for arm in record["arms"]:
        arm["settings"]["values"]["retrieval.maximum_candidates"] = 200
    first = Path(record["arms"][0]["reports"]["development"])
    second = Path(record["arms"][1]["reports"]["development"])
    for path, windows in ((first, [10, 20, 10, 20]), (second, [20, 10, 20, 10])):
        report = json.loads(path.read_text())
        for run, window in zip(report["runs"], windows, strict=True):
            run["rerank_window"] = window
        path.write_text(json.dumps(report), encoding="utf-8")
    table = render_comparison(record, split="development")
    assert "rerank_window=[10, 20]" in table
    assert "budgets differ or are incomplete" in table


def test_equal_schema_with_different_metric_definitions_is_refused(tmp_path):
    record = _two_split_record(tmp_path)
    path = Path(record["arms"][0]["reports"]["development"])
    report = json.loads(path.read_text())
    report["metric_definitions"] = {"exact_duplicate_slots": "unordered word sets"}
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ExperimentError, match="different metric definitions"):
        render_comparison(record)


def test_production_measurement_hash_also_authenticates_report(tmp_path):
    record = _two_split_record(tmp_path)
    arm = record["arms"][0]
    path = Path(arm["reports"]["development"])
    arm["measure"] = [
        {
            "split": "development",
            "report": str(path),
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ExperimentError, match="integrity check failed"):
        render_comparison(record)


def test_rerank_fallback_never_claims_confirmed_reranking(tmp_path):
    record = _two_split_record(tmp_path)
    path = Path(record["arms"][0]["reports"]["development"])
    report = json.loads(path.read_text())
    for run in report["runs"]:
        run["rerank_fallback"] = "model unavailable"
    report["degraded"] = {"rerank": "model unavailable"}
    path.write_text(json.dumps(report), encoding="utf-8")
    assert rows_for_run(record, MODE, "development")[0]["rerank_confirmed"] is False
    table = render_comparison(record, split="development")
    assert "DEGRADED" in table
    assert "differences from baseline" not in table


def test_a_missing_baseline_cannot_be_replaced_with_another_arm(tmp_path):
    record = _two_split_record(tmp_path)
    record["arms"][0]["reports"] = {}
    table = render_comparison(record)
    assert "NO DELTAS: the designated baseline" in table


def test_legacy_new_key_spellings_do_not_override_legacy_definitions(tmp_path):
    record = _redundancy_record(tmp_path)
    for arm in record["arms"]:
        path = Path(arm["report"])
        report = json.loads(path.read_text())
        report["schema_version"] = 2
        path.write_text(json.dumps(report), encoding="utf-8")
    rows = rows_for_run(record, MODE)
    for row in rows:
        assert row["mean_distinct_normalized_texts"] is None
        assert row["mean_lexical_containment_slots"] is None
        assert row["repeated_slot_rate_cross_target_family"] is None


def test_unknown_verdict_and_inconsistent_guard_do_not_claim_success(tmp_path):
    record = _two_split_record(tmp_path)
    record["verdict"] = "new-unrecognised-verdict"
    assert render_comparison(record).startswith("UNKNOWN RUN VERDICT")
    record["verdict"] = "verified"
    record["source_project"]["guard"]["unchanged"] = False
    assert render_comparison(record).startswith("NOT A VERIFIED MEASUREMENT")


def test_missing_latency_summary_does_not_trigger_toolkit_recalculation(tmp_path):
    record = _record(tmp_path, [("baseline", _row(), [1, 2, 3])])
    path = Path(record["arms"][0]["report"])
    report = json.loads(path.read_text())
    for key in ("p50_seconds", "p95_seconds"):
        report["summary"][MODE]["overall"].pop(key)
    path.write_text(json.dumps(report), encoding="utf-8")
    row = rows_for_run(record, MODE)[0]
    assert row["p50_seconds"] is None
    assert row["p95_seconds"] is None


def test_invalid_schema_is_a_named_refusal(tmp_path):
    record = _record(tmp_path, [("baseline", _row(), [1])])
    path = Path(record["arms"][0]["report"])
    report = json.loads(path.read_text())
    report["schema_version"] = "3"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ExperimentError, match="Unsupported report schema"):
        render_comparison(record)


def test_stage_denominators_and_unjudged_no_answer_scope_are_visible(tmp_path):
    record = _record(tmp_path, [("baseline", _row(), [1])])
    path = Path(record["arms"][0]["report"])
    report = json.loads(path.read_text())
    report["no_answer_support"] = "unjudged"
    report["summary"][MODE]["overall"].update(
        {
            "mean_target_present_fused_pre_rerank": None,
            "target_stage_evaluated_queries_fused_pre_rerank": 0,
            "mean_target_present_final": 0.7,
            "target_stage_evaluated_queries_final": 30,
        }
    )
    report["runs"][0]["collapsed_count"] = 4
    report["runs"][0]["candidate_count"] = 36
    path.write_text(json.dumps(report), encoding="utf-8")
    table = render_comparison(record)
    assert "fused_pre_rerank=- (0/30)" in table
    assert "final=0.700 (30/30)" in table
    assert 'no-answer scope: baseline: "unjudged"' in table
    assert "post_collapse_pool=[36] collapsed=[4]" in table


def test_prepared_record_is_not_presented_as_a_verified_measurement():
    record = {
        "verdict": "prepared",
        "source_project": {"guard": {"state": "unchanged"}},
        "arms": [],
    }
    text = render_comparison(record)
    assert text.startswith("PREPARATION ONLY: no measurements")
    assert "source guard unchanged" in text
    assert "No arm of this run reported a measured mode" in text


def test_shared_unjudged_no_answer_scope_is_concise_but_explicit(tmp_path):
    record = _two_split_record(tmp_path)
    for arm in record["arms"]:
        for path in arm["reports"].values():
            report = json.loads(Path(path).read_text())
            report["no_answer_support"] = {
                "measured": False,
                "judged_no_answer_queries": 0,
                "note": "long explanatory note",
            }
            Path(path).write_text(json.dumps(report), encoding="utf-8")
    text = render_comparison(record)
    assert text.count("no-answer scope: not measured;") == 2
    assert "every arm has 0 judged no-answer queries" in text
    assert "long explanatory note" not in text
