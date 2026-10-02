"""The one command's statuses, and the refusals it prints.

Every condition a reader must act on exits non-zero with the reason and the
command that fixes it. The status for a run whose source project changed is its
own value, because the arms ran and the corpus they claim to have measured is
not the one on disk, which is a different condition from a refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FIRST_GENERATION

from rag_experiments.cli import EXIT_SOURCE_CHANGED, EXIT_VERIFIED, main
from rag_experiments.report.record import RECORD_NAME, VERDICT_VERIFIED
from rag_experiments.sandbox import create, list_sandboxes, snapshot


def test_no_arguments_prints_the_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_VERIFIED
    assert "rag-experiments" in capsys.readouterr().out


def test_the_version_names_the_engine_the_environment_points_at(
    capsys: pytest.CaptureFixture[str],
    app_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_RAG_APP_SOURCE", str(app_source))
    assert main(["--version"]) == EXIT_VERIFIED
    out = capsys.readouterr().out
    assert "rag-experiments" in out
    assert str(app_source) in out
    assert "no revision" not in out


def test_the_version_names_its_own_remedy_when_there_is_no_engine(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RESEARCH_RAG_APP_SOURCE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["--version"]) == EXIT_VERIFIED
    out = capsys.readouterr().out
    assert "engine: not found" in out
    assert "--app-source" in out
    assert "RESEARCH_RAG_APP_SOURCE" in out


def test_inspect_reports_what_a_copy_would_carry(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", "--project", str(project)]) == EXIT_VERIFIED
    out = capsys.readouterr().out
    assert FIRST_GENERATION in out
    assert "project lock" in out
    assert "never written" in out
    assert "account's project registry" in out


def test_inspect_refuses_something_that_is_not_a_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inspect", "--project", str(tmp_path)]) == 1
    assert "not a research-rag project" in capsys.readouterr().err


def test_verify_prints_a_digest_and_writes_nothing(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = snapshot(project).digest()
    assert main(["verify", "--project", str(project)]) == EXIT_VERIFIED
    out = capsys.readouterr().out
    assert before in out
    assert snapshot(project).digest() == before


def test_verify_against_the_same_digest_says_unchanged(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    digest = snapshot(project).digest()
    assert (
        main(["verify", "--project", str(project), "--expect", digest]) == EXIT_VERIFIED
    )
    assert "unchanged" in capsys.readouterr().out


def test_verify_against_another_digest_has_its_own_status(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["verify", "--project", str(project), "--expect", "0" * 64])
        == EXIT_SOURCE_CHANGED
    )
    assert "is not the one with digest" in capsys.readouterr().err


def test_a_sandbox_is_created_listed_and_removed(
    project: Path, workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "sandbox",
                "create",
                "--project",
                str(project),
                "--workspace",
                str(workspace),
                "--name",
                "one",
            ]
        )
        == EXIT_VERIFIED
    )
    out = capsys.readouterr().out
    assert str(workspace / "one" / "project") in out
    assert FIRST_GENERATION in out

    assert main(["sandbox", "list", "--workspace", str(workspace)]) == EXIT_VERIFIED
    assert "one" in capsys.readouterr().out

    assert (
        main(["sandbox", "remove", "--workspace", str(workspace), "--name", "one"])
        == EXIT_VERIFIED
    )
    assert "removed" in capsys.readouterr().out
    assert list_sandboxes(workspace) == []


def test_creating_a_sandbox_leaves_the_source_alone(
    project: Path, workspace: Path
) -> None:
    before = snapshot(project)
    assert (
        main(
            [
                "sandbox",
                "create",
                "--project",
                str(project),
                "--workspace",
                str(workspace),
                "--name",
                "one",
            ]
        )
        == EXIT_VERIFIED
    )
    assert snapshot(project).digest() == before.digest()


def test_sandbox_with_no_subcommand_is_a_refusal(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sandbox"]) == 1
    assert "create, list, or remove" in capsys.readouterr().err


def test_removing_a_directory_this_harness_did_not_make_is_refused(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / "stranger").mkdir()
    assert (
        main(["sandbox", "remove", "--workspace", str(workspace), "--name", "stranger"])
        == 1
    )
    assert "not a sandbox this harness made" in capsys.readouterr().err
    assert (workspace / "stranger").exists()


def test_a_dry_run_prepares_every_arm_and_measures_nothing(
    make_spec,
    project: Path,
    workspace: Path,
    app_source: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = make_spec(
        [
            {"name": "baseline", "kind": "settings", "overlay": {}},
            {
                "name": "deeper",
                "kind": "settings",
                "overlay": {"retrieval.maximum_candidates": 80},
            },
        ]
    )
    before = snapshot(project).digest()
    status = main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--app-source",
            str(app_source),
            "--workspace",
            str(workspace),
            "--runs",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert status == EXIT_VERIFIED
    assert "arm baseline" in out
    assert "arm deeper" in out
    assert "Nothing was measured" in out
    # No report is written and no run record exists: a dry run is a check, not a
    # measurement, and a reader must not find a record claiming otherwise.
    assert not list((tmp_path / "runs").rglob(RECORD_NAME))
    assert snapshot(project).digest() == before


def test_a_dry_run_refuses_an_unknown_setting_before_measuring(
    make_spec,
    workspace: Path,
    app_source: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = make_spec(
        [{"name": "bad", "kind": "settings", "overlay": {"retrieval.no_such_knob": 1}}]
    )
    status = main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--app-source",
            str(app_source),
            "--workspace",
            str(workspace),
            "--runs",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )
    assert status == 1
    assert "retrieval.no_such_knob" in capsys.readouterr().err


def test_a_specification_naming_no_judged_set_is_refused(
    tmp_path: Path,
    workspace: Path,
    app_source: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "x",
                "source_project": str(tmp_path),
                "judgments": "absent.json",
                "arms": [{"name": "a", "kind": "settings", "overlay": {}}],
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "run",
                "--spec",
                str(spec_path),
                "--app-source",
                str(app_source),
                "--workspace",
                str(workspace),
                "--runs",
                str(tmp_path / "runs"),
            ]
        )
        == 1
    )
    assert "judged set does not exist" in capsys.readouterr().err


def test_an_app_source_that_is_not_an_engine_is_refused(
    make_spec,
    tmp_path: Path,
    workspace: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = main(
        [
            "run",
            "--spec",
            str(make_spec([{"name": "a", "kind": "settings", "overlay": {}}])),
            "--app-source",
            str(tmp_path),
            "--workspace",
            str(workspace),
            "--runs",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )
    assert status == 1
    assert "research-rag tree" in capsys.readouterr().err


def test_compare_prints_a_record_it_reads(
    project: Path, workspace: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create(workspace, name="one", source=project)
    run = tmp_path / "run"
    run.mkdir()
    (run / RECORD_NAME).write_text(
        json.dumps(
            {
                "verdict": VERDICT_VERIFIED,
                "source_project": {
                    "guard": {
                        "unchanged": True,
                        "differences": [],
                        "before": {
                            "file_count": 5,
                            "byte_count": 100,
                            "digest": "c" * 64,
                        },
                    }
                },
                "arms": [],
            }
        ),
        encoding="utf-8",
    )
    assert main(["compare", "--run", str(run)]) == EXIT_VERIFIED
    assert "source project unchanged" in capsys.readouterr().out


def test_compare_json_prints_the_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / RECORD_NAME).write_text('{"verdict": "verified"}', encoding="utf-8")
    assert main(["compare", "--run", str(run), "--json"]) == EXIT_VERIFIED
    assert json.loads(capsys.readouterr().out)["verdict"] == "verified"


def test_compare_of_a_directory_with_no_record_is_a_condition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["compare", "--run", str(tmp_path / "absent")]) == 1
    assert RECORD_NAME in capsys.readouterr().err


def test_a_run_record_whose_verdict_is_not_verified_exits_with_its_own_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / RECORD_NAME).write_text(
        json.dumps(
            {
                "verdict": "source_project_changed",
                "source_project": {
                    "guard": {"unchanged": False, "differences": [], "before": {}}
                },
                "arms": [],
            }
        ),
        encoding="utf-8",
    )
    assert main(["compare", "--run", str(run)]) == EXIT_SOURCE_CHANGED
    assert "NOT A MEASUREMENT" in capsys.readouterr().out
