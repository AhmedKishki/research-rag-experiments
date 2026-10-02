"""The one command, and the four jobs it does.

Every subcommand is a verb on an experiment or on the state one left behind. The
command line reads; it never writes a research-rag project, never computes a
quality metric, and never decides that a run succeeded. A condition a reader must
act on exits non-zero with the reason and the command that fixes it, which is the
same rule the app under test follows so the two reports read alike.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import toolkit_version
from .engine.locate import EngineSource, locate_engine
from .errors import ExperimentError
from .experiment import load_spec
from .report import compare, read_record
from .report.record import VERDICT_SOURCE_CHANGED
from .run import run_experiment
from .sandbox import create as create_sandbox
from .sandbox import list_sandboxes, read_layout, selected_generation, snapshot
from .sandbox import remove as remove_sandbox

#: What the command is called, matching the distribution and the entry point.
PROGRAM = "rag-experiments"

#: Where a run's sandboxes go unless the caller says otherwise, relative to the
#: current directory. A sandbox is a large copy of a corpus, so the default is a
#: directory a reader can see is disposable.
DEFAULT_WORKSPACE = Path("workspaces")

#: Where run records go unless the caller says otherwise.
DEFAULT_RUNS = Path("runs")

#: The variable that names the engine under test for a machine that keeps its
#: research-rag checkout somewhere this repository cannot know. It is the same name
#: the tests use, and it is read by `--version` so the answer is the engine a run
#: would measure rather than a guess from the working directory.
APP_SOURCE_ENV = "RESEARCH_RAG_APP_SOURCE"

#: The exit status a run whose source project changed carries. It is not 1,
#: because 1 is a refusal and a run that measured nothing against a moved corpus
#: is a different condition: the arms ran, and the corpus they claim to have
#: measured is not the one that exists.
EXIT_SOURCE_CHANGED = 3

#: The exit status for a run that completed and left the source project alone.
EXIT_VERIFIED = 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Measure research-rag retrieval in an isolated copy of a project. "
            "An experiment never writes inside the project it measures, and a "
            "run that changed it says so instead of reporting numbers."
        ),
        epilog=(
            "Start with: "
            f"{PROGRAM} inspect --project PATH, then {PROGRAM} run --spec FILE.json"
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print this toolkit's version and the engine it will find.",
    )
    commands = parser.add_subparsers(dest="command")

    inspect = commands.add_parser(
        "inspect",
        help="Say what a project holds and what a run would copy from it.",
    )
    inspect.add_argument(
        "--project", type=Path, required=True, help="The project root to read."
    )

    sandbox = commands.add_parser(
        "sandbox", help="Create, list, and remove the disposable copies."
    )
    sandbox_commands = sandbox.add_subparsers(dest="sandbox_command")

    make = sandbox_commands.add_parser(
        "create", help="Copy a project into a workspace, and print where it went."
    )
    make.add_argument("--project", type=Path, required=True)
    make.add_argument(
        "--workspace", type=Path, default=DEFAULT_WORKSPACE, help="Where the copy goes."
    )
    make.add_argument("--name", required=True, help="One path segment naming the copy.")
    make.add_argument(
        "--generation",
        action="append",
        default=[],
        help="A generation to copy. Repeatable. Default: the one the project selects.",
    )

    listing = sandbox_commands.add_parser(
        "list", help="List the copies in a workspace."
    )
    listing.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)

    drop = sandbox_commands.add_parser(
        "remove", help="Delete a copy this harness made."
    )
    drop.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    drop.add_argument("--name", required=True)

    guard = commands.add_parser(
        "verify",
        help=(
            "Digest a project and, with --expect, compare it to a digest printed "
            "earlier. Writes nothing."
        ),
    )
    guard.add_argument("--project", type=Path, required=True)
    guard.add_argument("--expect", help="A digest to compare against.")

    run = commands.add_parser(
        "run", help="Measure every arm of a specification and write one record."
    )
    run.add_argument("--spec", type=Path, required=True, help="The run specification.")
    run.add_argument(
        "--app-source",
        type=Path,
        required=True,
        help=(
            "The research-rag tree to measure. Its revision is named in the "
            "record, and a code arm is copied from it."
        ),
    )
    run.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    run.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    run.add_argument(
        "--no-keep-sandboxes",
        action="store_true",
        help="Delete each sandbox once its arm has been measured.",
    )
    run.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "Measure without the app's validation pass first. A judged target "
            "that cannot be resolved then costs an hour of inference before it "
            "is found."
        ),
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Materialize each arm's sandbox, pin its settings, and stop. This is "
            "the check to run before an expensive run."
        ),
    )

    show = commands.add_parser("compare", help="Print a run's arms side by side.")
    show.add_argument("--run", type=Path, required=True, help="A run directory.")
    show.add_argument(
        "--mode",
        action="append",
        default=[],
        help="A mode to tabulate. Repeatable. Default: every mode the run reported.",
    )
    show.add_argument(
        "--split",
        help="One judged split to tabulate. Default: every split the run reported.",
    )
    show.add_argument("--json", action="store_true", help="Print the record instead.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command, and turn a condition into a status and one message."""

    parser = _parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False) and not args.command:
        return _version()
    if not args.command:
        parser.print_help()
        return EXIT_VERIFIED
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "sandbox":
            return _sandbox(args)
        if args.command == "verify":
            return _verify(args)
        if args.command == "run":
            return _run(args)
        if args.command == "compare":
            return _compare(args)
    except ExperimentError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return EXIT_VERIFIED


def _version() -> int:
    """Print this toolkit's version and the engine a run would measure.

    The engine is the one the environment names, or the working directory if that
    is itself a checkout. Saying "not found" with the flag that fixes it is more
    useful than a path that happens to be a directory.
    """

    print(f"{PROGRAM} {toolkit_version()}")
    named = os.environ.get(APP_SOURCE_ENV, "").strip()
    try:
        engine = locate_engine(Path(named) if named else Path.cwd())
    except ExperimentError:
        print(
            "engine: not found; pass --app-source, or set "
            f"{APP_SOURCE_ENV} at a research-rag checkout"
        )
        return EXIT_VERIFIED
    described = engine.describe()
    state = "clean" if described["dirty"] is False else "dirty"
    print(
        f"engine: {described['revision'] or 'no revision'} "
        f"({described['branch'] or 'no branch'}, {state}) at {described['root']}"
    )
    return EXIT_VERIFIED


def _inspect(args: argparse.Namespace) -> int:
    """Say what a project holds, and what a copy of it would carry."""

    layout = read_layout(args.project)
    pointed = selected_generation(layout)
    from .sandbox import existing_generations

    others = [item for item in existing_generations(layout) if item != pointed]
    lines = [
        ("project root", layout.root),
        ("originals", layout.source_root),
        ("review state", layout.portable_root),
        ("runtime", layout.state_root),
        ("selected generation", pointed),
        ("other generations", ", ".join(others) if others else "none"),
    ]
    width = max(len(label) for label, _ in lines) + 2
    for label, value in lines:
        print(f"{label + ':':<{width}}{value}")
    print()
    print(
        "copied runtime:   the selected-generation pointer and the named "
        "generations only; process state, a project lock, and a build journal "
        "are not carried"
    )
    print("never written:    the project above, and the account's project registry")
    return EXIT_VERIFIED


def _sandbox(args: argparse.Namespace) -> int:
    if not args.sandbox_command:
        print(f"{PROGRAM} sandbox: say create, list, or remove", file=sys.stderr)
        return 1
    if args.sandbox_command == "create":
        sandbox = create_sandbox(
            args.workspace,
            name=args.name,
            source=args.project,
            generations=tuple(args.generation),
        )
        print(f"sandbox: {sandbox.root}")
        print(f"  project id: {sandbox.project_id}")
        print(f"  generations: {', '.join(sandbox.generation_ids)}")
        print(
            f"  copied {sandbox.file_count} files, {sandbox.byte_count} bytes in "
            f"{sandbox.elapsed_seconds:.1f} s"
        )
        print(f"  record: {sandbox.record_path}")
        print(
            "  settings are not pinned yet; a run pins them, and "
            "`rag-experiments run --dry-run` will do that without measuring"
        )
        return EXIT_VERIFIED
    if args.sandbox_command == "list":
        found = list_sandboxes(args.workspace)
        if not found:
            print(f"no sandboxes in {args.workspace}")
            return EXIT_VERIFIED
        print(f"{'name':<28}{'project id':<40}{'generations':<12}bytes")
        for entry in found:
            print(
                f"{entry.get('name', '?'):<28}{entry.get('project_id', '?'):<40}"
                f"{len(entry.get('generation_ids') or []):<12}"
                f"{entry.get('byte_count', 0)}"
            )
        return EXIT_VERIFIED
    removed = remove_sandbox(args.workspace, args.name)
    print(f"removed {removed}")
    return EXIT_VERIFIED


def _verify(args: argparse.Namespace) -> int:
    """Digest a project, and say whether it matches a digest printed before.

    This is the standalone form of the guard a run takes on its own: it is what a
    reader runs to confirm a project is byte-identical after work that was not
    done through this harness.
    """

    taken = snapshot(args.project)
    described = taken.describe()
    if not args.expect:
        print(f"{described['file_count']} files, {described['byte_count']} bytes")
        print(f"digest: {described['digest']}")
        print(f"took {described['elapsed_seconds']} s")
        return EXIT_VERIFIED
    if taken.digest() == args.expect:
        print(f"unchanged: {described['digest']}")
        return EXIT_VERIFIED
    print(
        f"{PROGRAM}: this project is not the one with digest {args.expect}; it is "
        f"{described['digest']}",
        file=sys.stderr,
    )
    return EXIT_SOURCE_CHANGED


def _run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    engine = locate_engine(args.app_source)
    if args.dry_run:
        return _dry_run(spec, engine, args)

    outcome = run_experiment(
        spec,
        app_source=engine,
        workspace=args.workspace,
        runs_directory=args.runs,
        keep_sandboxes=not args.no_keep_sandboxes,
        validate_first=not args.no_validate,
    )
    print(outcome.comparison)
    print()
    print(f"record: {outcome.record_path}")
    print(f"verdict: {outcome.verdict}")
    _report_validation(outcome)
    if outcome.verdict == VERDICT_SOURCE_CHANGED:
        guard = outcome.record["source_project"]["guard"]
        print(
            f"{PROGRAM}: the source project changed during this run, so its numbers "
            f"do not describe the corpus on disk: "
            f"{', '.join(guard['differences'][:8])}",
            file=sys.stderr,
        )
        return EXIT_SOURCE_CHANGED
    return EXIT_VERIFIED


def _report_validation(outcome: Any) -> None:
    """Say whether every judged target resolved, per arm and per split.

    The plan asks for every target-resolution failure to be logged rather than
    summarised away, so an arm that resolved everything says so and an arm that
    did not is named with the log that holds the message.
    """

    for result in outcome.results:
        for item in result.validate:
            if item.exit_code == 0:
                print(
                    f"validate {result.arm.name} / {item.split}: "
                    f"every judged target resolved"
                )
            else:
                print(
                    f"validate {result.arm.name} / {item.split}: exit "
                    f"{item.exit_code}; see {result.log_path}"
                )


def _dry_run(spec: Any, engine: EngineSource, args: argparse.Namespace) -> int:
    """Materialize and pin every arm, then stop before anything is measured.

    This is the check that answers "can this run start?" without spending an
    hour. It is the same preparation a real run performs, so what it refuses is
    what a real run would refuse: a missing original, a missing generation, an
    unknown setting, an out-of-range value, or a patch that does not apply.
    """

    from .experiment.execute import prepare_arm

    workspace = Path(args.workspace).expanduser().resolve()
    runs = Path(args.runs).expanduser().resolve() / "dry-run"
    for arm in spec.arms:
        prepared = prepare_arm(
            spec,
            arm,
            workspace=workspace,
            run_directory=runs,
            app_source=engine,
        )
        print(f"arm {arm.name}: {prepared.sandbox.root}")
        print(f"  engine: {prepared.engine.root}")
        print(
            f"  settings: {prepared.settings['key_count']} pinned, "
            f"{len(prepared.settings['overridden'])} overridden"
        )
        print(f"  generations: {', '.join(prepared.sandbox.generation_ids)}")
        if prepared.checkout is not None:
            print(
                f"  checkout: base {prepared.checkout.base_revision}, diff "
                f"{len(prepared.checkout.resulting_diff)} bytes"
            )
    print()
    print(
        "dry run complete. Nothing was measured and the source project was not written."
    )
    return EXIT_VERIFIED


def _compare(args: argparse.Namespace) -> int:
    record = read_record(args.run)
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return EXIT_VERIFIED
    print(compare.render_comparison(record, tuple(args.mode), args.split))
    verdict = str(record.get("verdict") or "")
    if verdict == VERDICT_SOURCE_CHANGED:
        return EXIT_SOURCE_CHANGED
    return EXIT_VERIFIED


if __name__ == "__main__":
    raise SystemExit(main())
