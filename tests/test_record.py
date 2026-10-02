"""That a run's verdict follows from the guard, and that the record loads.

The verdict is the field a reader checks before believing a number, so it is
computed from the guard, the engine's own digest, and the arms that measured, and
from nothing else. `verified` is the narrowest state here, so these tests also hold
what disqualifies a run: a corpus that moved, a corpus that could not be checked, a
failure, and a run that measured nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_experiments.report.record import (
    GUARD_UNKNOWN,
    RECORD_NAME,
    RECORD_SCHEMA_VERSION,
    VERDICT_ENGINE_CHANGED,
    VERDICT_FAILED,
    VERDICT_INCOMPLETE,
    VERDICT_INTERRUPTED,
    VERDICT_NO_ARMS,
    VERDICT_SOURCE_CHANGED,
    VERDICT_VERIFIED,
    SourceGuard,
    close_guard,
    read_record,
    take_guard,
    write_record,
)

#: An engine that did not move, which is what `verified` additionally requires.
STILL_ENGINE = {
    "content_sha256": "d" * 64,
    "content_sha256_after": "d" * 64,
    "changed": False,
}


def _measured(name: str) -> dict[str, object]:
    return {"arm": {"name": name}, "measured": True}


def _write(
    directory: Path,
    *,
    arms: list[dict[str, object]] | None = None,
    guard: SourceGuard,
    engine: dict[str, object] | None = None,
    error: str | None = None,
    interruption: str | None = None,
) -> Path:
    listed = [_measured(f"arm{index}") for index in range(1)] if arms is None else arms
    return write_record(
        directory,
        name="fixture",
        elapsed_seconds=1.5,
        spec={"name": "fixture"},
        engine=dict(STILL_ENGINE if engine is None else engine),
        judgments=[{"path": "/tmp/judged.json", "sha256": "b" * 64}],
        source={"project_root": "/tmp/corpus"},
        arms=listed,
        guard=guard,
        started="2026-01-01T00:00:00.000000Z",
        error=error,
        interruption=interruption,
    )


def _verdict(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["verdict"])


def test_a_run_that_left_the_source_alone_is_verified(
    project: Path, tmp_path: Path
) -> None:
    before = take_guard(project)
    guard = close_guard(before)
    path = _write(tmp_path / "run", guard=guard)
    assert _verdict(path) == VERDICT_VERIFIED
    assert guard.unchanged
    assert guard.state == "unchanged"


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
    assert len(record["arms"]) == 1


def test_a_changed_source_outranks_every_other_condition(
    project: Path, tmp_path: Path
) -> None:
    before = take_guard(project)
    (project / "sources" / "a-book.pdf").write_bytes(b"%PDF-1.7\nchanged")
    guard = close_guard(before)
    path = _write(
        tmp_path / "run",
        arms=[_measured("arm0"), {"arm": {"name": "arm1"}, "measured": False}],
        guard=guard,
        engine={**STILL_ENGINE, "changed": True, "content_sha256_after": "e" * 64},
        error="arm1 refused: exit 2",
    )
    assert _verdict(path) == VERDICT_SOURCE_CHANGED


def test_a_run_with_no_armed_measurement_says_so(project: Path, tmp_path: Path) -> None:
    guard = close_guard(take_guard(project))
    path = _write(tmp_path / "run", arms=[], guard=guard)
    assert _verdict(path) == VERDICT_NO_ARMS


def test_a_run_that_stopped_before_measuring_anything_failed(
    project: Path, tmp_path: Path
) -> None:
    # Zero successes is still a record: the arms that were attempted, the command
    # that refused, and the error are all worth more than an absent file.
    guard = close_guard(take_guard(project))
    path = _write(
        tmp_path / "run",
        arms=[{"arm": {"name": "arm0"}, "measured": False, "failure": "exit 2"}],
        guard=guard,
        error="Arm 'arm0' failed measuring split 'development' with exit 2.",
    )
    assert _verdict(path) == VERDICT_FAILED


def test_a_run_with_one_failed_arm_is_incomplete(project: Path, tmp_path: Path) -> None:
    guard = close_guard(take_guard(project))
    path = _write(
        tmp_path / "run",
        arms=[_measured("arm0"), {"arm": {"name": "arm1"}, "measured": False}],
        guard=guard,
        error="Arm 'arm1' failed measuring split 'held-out' with exit 2.",
    )
    assert _verdict(path) == VERDICT_INCOMPLETE


def test_an_interrupted_run_is_its_own_verdict(project: Path, tmp_path: Path) -> None:
    guard = close_guard(take_guard(project))
    path = _write(
        tmp_path / "run",
        arms=[_measured("arm0")],
        guard=guard,
        interruption="KeyboardInterrupt: ",
    )
    assert _verdict(path) == VERDICT_INTERRUPTED


def test_an_engine_that_moved_under_the_run_disqualifies_it(
    project: Path, tmp_path: Path
) -> None:
    guard = close_guard(take_guard(project))
    path = _write(
        tmp_path / "run",
        guard=guard,
        engine={
            "content_sha256": "d" * 64,
            "content_sha256_after": "e" * 64,
            "changed": True,
        },
    )
    assert _verdict(path) == VERDICT_ENGINE_CHANGED


def test_an_engine_that_could_not_be_digest_after_the_run_is_not_verified(
    project: Path, tmp_path: Path
) -> None:
    guard = close_guard(take_guard(project))
    path = _write(
        tmp_path / "run",
        guard=guard,
        engine={"content_sha256": None, "content_sha256_after": None, "changed": None},
    )
    assert _verdict(path) == VERDICT_FAILED


def test_a_corpus_that_could_not_be_checked_is_unknown_not_unchanged(
    tmp_path: Path,
) -> None:
    # No digest at all, neither before nor after: the harness promised the corpus
    # was untouched and did not get to find out.
    guard = close_guard(None)
    assert guard.state == GUARD_UNKNOWN
    assert not guard.unchanged
    assert guard.describe()["before"] is None


def test_the_record_states_the_digest_it_verified_against(
    project: Path, tmp_path: Path
) -> None:
    before = take_guard(project)
    path = _write(tmp_path / "run", guard=close_guard(before))
    guard = json.loads(path.read_text(encoding="utf-8"))["source_project"]["guard"]
    assert guard["before"]["digest"] == before.digest()
    assert guard["after"]["digest"] == before.digest()
    # The record names what the guard watched and what it did not, rather than
    # asking the reader to trust that everything else was covered.
    assert all(
        path.startswith(".research-rag/runtime/")
        for path in guard["before"]["excluded_paths"]
    )
    assert guard["state"] == "unchanged"


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
    assert record["run"]["complete"] is True
    assert record["measured_arm_count"] == 1
    assert record["toolkit_version"]


def test_a_record_names_the_failure_it_carries(project: Path, tmp_path: Path) -> None:
    guard = close_guard(take_guard(project))
    path = _write(
        tmp_path / "run",
        arms=[{"arm": {"name": "arm0"}, "measured": False}],
        guard=guard,
        error="Arm 'arm0' refused.",
    )
    record = read_record(tmp_path / "run")
    assert record["error"] == "Arm 'arm0' refused."
    assert record["run"]["complete"] is False
    assert record["arms"][0]["arm"]["name"] == "arm0"
    assert _verdict(path) == VERDICT_FAILED


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
