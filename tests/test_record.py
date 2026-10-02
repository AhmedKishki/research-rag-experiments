"""That a run's verdict follows from the guard, and that the record loads.

The verdict is the field a reader checks before believing a number, so it is
computed from the two digests and from nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_experiments.report.record import (
    RECORD_NAME,
    RECORD_SCHEMA_VERSION,
    VERDICT_NO_ARMS,
    VERDICT_SOURCE_CHANGED,
    VERDICT_VERIFIED,
    close_guard,
    read_record,
    take_guard,
    write_record,
)


def _write(
    directory: Path,
    *,
    arms: int = 1,
    guard: object,
) -> Path:
    return write_record(
        directory,
        name="fixture",
        elapsed_seconds=1.5,
        spec={"name": "fixture"},
        engine={"revision": "a" * 40},
        judgments={"path": "/tmp/judged.json", "sha256": "b" * 64},
        source={"project_root": "/tmp/corpus"},
        arms=[{"arm": {"name": f"arm{index}"}} for index in range(arms)],
        guard=guard,
        started="2026-01-01T00:00:00.000000Z",
    )


def test_a_run_that_left_the_source_alone_is_verified(
    project: Path, tmp_path: Path
) -> None:
    before = take_guard(project)
    guard = close_guard(before)
    path = _write(tmp_path / "run", guard=guard)
    assert json.loads(path.read_text(encoding="utf-8"))["verdict"] == VERDICT_VERIFIED
    assert guard.unchanged


def test_a_run_that_changed_the_source_is_not_a_measurement(
    project: Path, tmp_path: Path
) -> None:
    before = take_guard(project)
    (project / "sources" / "a-book.pdf").write_bytes(b"%PDF-1.7\nchanged")
    guard = close_guard(before)
    path = _write(tmp_path / "run", guard=guard)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["verdict"] == VERDICT_SOURCE_CHANGED
    assert record["source_project"]["guard"]["differences"] == [
        "changed: sources/a-book.pdf"
    ]
    # The record is still written: a failure with its provenance is the useful
    # thing, and the numbers beside it are disqualified rather than hidden.
    assert record["arms"] == [{"arm": {"name": "arm0"}}]


def test_a_run_with_no_armed_measurement_says_so(project: Path, tmp_path: Path) -> None:
    guard = close_guard(take_guard(project))
    path = _write(tmp_path / "run", arms=0, guard=guard)
    assert json.loads(path.read_text(encoding="utf-8"))["verdict"] == VERDICT_NO_ARMS


def test_the_record_states_the_digest_it_verified_against(
    project: Path, tmp_path: Path
) -> None:
    before = take_guard(project)
    path = _write(tmp_path / "run", guard=close_guard(before))
    guard = json.loads(path.read_text(encoding="utf-8"))["source_project"]["guard"]
    assert guard["before"]["digest"] == before.digest()
    assert guard["after"]["digest"] == before.digest()
    assert guard["guarded_entries"] == [".research-rag", "sources"]


def test_a_record_reads_back_and_names_its_schema(
    project: Path, tmp_path: Path
) -> None:
    guard = close_guard(take_guard(project))
    directory = tmp_path / "run"
    _write(directory, guard=guard)
    record = read_record(directory)
    assert record["schema_version"] == RECORD_SCHEMA_VERSION
    assert record["run"]["name"] == "fixture"
    assert record["run"]["directory"] == str(directory)
    assert record["toolkit_version"]


def test_reading_a_directory_with_no_record_is_a_condition(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as caught:
        read_record(tmp_path / "absent")
    assert RECORD_NAME in str(caught.value)
    assert "rag-experiments run" in str(caught.value)


def test_the_two_digests_are_taken_around_the_work_not_together(
    project: Path,
) -> None:
    before = take_guard(project)
    (project / "sources" / "a-book.pdf").write_bytes(b"%PDF-1.7\nchanged")
    after = take_guard(project)
    assert before.digest() != after.digest()


def test_a_digest_names_the_root_it_came_from(project: Path) -> None:
    assert take_guard(project).describe()["root"] == str(project)
