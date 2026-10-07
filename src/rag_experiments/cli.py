"""The one command.

Every command here is a status, a refusal with the reason, or the name of a file a
reader should open. The statuses are the record's verdicts, so a caller can tell a
run that measured nothing from one that refused, and neither is ever reported as
the exit code for a completed measurement.

A failed run exits non-zero and prints where its record is. That is the whole
point of writing a record on every exit: the numbers that did arrive, the command
that stopped the run, and the log that holds the output are all one path away.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import toolkit_version
from .engine.locate import EngineSource, locate_engine
from .errors import ExperimentError
from .experiment import RunSpec, load_spec
from .report import VERDICT_ENGINE_CHANGED, VERDICT_SOURCE_CHANGED
from .report.compare import render_comparison
from .report.record import (
    RECORD_NAME,
    VERDICT_FAILED,
    VERDICT_INCOMPLETE,
    VERDICT_INTERRUPTED,
    VERDICT_NO_ARMS,
    VERDICT_PREPARED,
    VERDICT_VERIFIED,
    read_record,
)
from .run import RunOutcome, prepare_experiment, run_experiment
from .sandbox import (
    create,
    list_sandboxes,
    read_layout,
    remove,
    snapshot,
    volatile_paths,
)
from .sandbox import selected_generation as pointed_generation

PROGRAM = "rag-experiments"

#: The variable that names the engine under test for a machine that keeps its
#: research-rag checkout somewhere this repository cannot know. `--version` reads
#: it, so the answer is the engine a run would measure rather than a guess from the
#: working directory.
APP_SOURCE_ENV = "RESEARCH_RAG_APP_SOURCE"

#: A run whose source project's bytes moved is not a measurement, whatever its
#: numbers say, and it has its own status so a caller can tell that apart from a
#: run that refused to start.
EXIT_SOURCE_CHANGED = 3

#: A run whose engine files moved under it is the same kind of disqualification and
#: is likewise its own status.
EXIT_ENGINE_CHANGED = 4

#: A run that stopped part-way: some arms measured, some did not.
EXIT_INCOMPLETE = 2

#: The exit code for a run that completed its measurement, for a preparation run
#: that resolved every arm and left the source project alone, and for every command
#: whose success is the absence of a problem.
EXIT_VERIFIED = 0

#: The exit code for a refusal, a failed run, and a run that measured nothing. These
#: are conditions a reader acts on rather than measurements they can trust.
EXIT_REFUSED = 1


def main(argv: list[str] | None = None) -> int:
    # Parent-side layout/registry inspection also imports the editable app.
    # Child-only suppression still writes caches during a cold CLI inspection.
    sys.dont_write_bytecode = True
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        return _version()
    if args.command is None:
        # No command is not a refusal: the help is the answer, and it is what a
        # reader who typed the command name alone is asking for.
        parser.print_help()
        return EXIT_VERIFIED
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "sandbox":
            return _sandbox(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "annotation":
            return _annotation(args)
        if args.command == "performance":
            return _performance(args)
        if args.command == "verify":
            return _verify(args)
        return _inspect(args)  # inspect: the default command when one is named
    except (ExperimentError, FileNotFoundError, ValueError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Measure research-rag retrieval arms against a project, in disposable "
            "copies, and leave one record per run."
        ),
    )
    parser.add_argument(
        "--version", action="store_true", help="Print version and exit."
    )
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="Measure every arm of a specification.")
    # `run` takes no `--project`: the specification names the project it measures,
    # and a second source of truth for it would be one more thing to disagree.
    _add_app_source_argument(run)
    run.add_argument(
        "--spec", type=Path, required=True, help="Run specification JSON file."
    )
    run.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Directory the run's sandboxes are made in.",
    )
    run.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="Directory the run's own directory is created under.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare every arm without searching and retain a guarded preparation record.",
    )
    run.add_argument(
        "--no-keep-sandboxes",
        dest="keep_sandboxes",
        action="store_false",
        help=(
            "Remove each arm's sandbox when its measurement succeeds. Reports, "
            "logs, and checkouts stay in the run's directory."
        ),
    )
    run.add_argument(
        "--no-validate",
        dest="validate_first",
        action="store_false",
        help=(
            "Skip the harness's own validation pass, which resolves every judged "
            "target before any search runs."
        ),
    )

    verify = commands.add_parser(
        "verify", help="Digest a project and compare the digest."
    )
    _add_source_arguments(verify)
    verify.add_argument(
        "--expect",
        help=(
            "Digest to compare against. Omit it to print the project's digest "
            "without writing anything."
        ),
    )

    inspect = commands.add_parser(
        "inspect", help="Report what a run would copy, without copying it."
    )
    _add_source_arguments(inspect)

    performance = commands.add_parser(
        "performance",
        help="Run cold-process and warm-search probes without quality scoring.",
    )
    performance.add_argument("--spec", type=Path, required=True)
    performance.add_argument("--app-source", type=Path, required=True)
    performance.add_argument("--workspace", type=Path, required=True)
    performance.add_argument("--runs", type=Path, required=True)
    performance.add_argument("--blocks", type=int, default=3)
    performance.add_argument("--warmup-count", type=int, default=3)
    performance.add_argument("--seed", type=int, default=0)

    compare = commands.add_parser(
        "compare", help="Print the comparison table for a run."
    )
    compare.add_argument(
        "--run", type=Path, required=True, help="Directory holding a run record."
    )
    compare.add_argument(
        "--json", action="store_true", help="Print the record instead."
    )
    compare.add_argument(
        "--modes", default="", help="Comma-separated modes to tabulate."
    )
    compare.add_argument("--split", default=None, help="One split to tabulate.")

    annotation = commands.add_parser(
        "annotation",
        help="Prepare blinded private author-review packets and validate judgments.",
    )
    annotation_commands = annotation.add_subparsers(
        dest="annotation_command", required=True
    )
    build = annotation_commands.add_parser(
        "build", help="Pool verified candidates; assign no judgments."
    )
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument(
        "--output", type=Path, required=True, help="New private package directory."
    )
    check = annotation_commands.add_parser(
        "check", help="Validate returned author judgments."
    )
    check.add_argument("--pool", type=Path, required=True)
    check.add_argument("--judgments", type=Path, required=True)
    check.add_argument("--require-complete", action="store_true")
    check.add_argument(
        "--details",
        action="store_true",
        help="Include every pending/uncertain opaque identifier.",
    )
    export = annotation_commands.add_parser(
        "export",
        help="Join completed author labels to retained rankings; do not score them.",
    )
    export.add_argument("--pool", type=Path, required=True)
    export.add_argument("--judgments", type=Path, required=True)
    export.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New private handoff JSON outside the frozen packet.",
    )
    score = annotation_commands.add_parser(
        "score",
        help="Delegate a completed handoff to the app-owned saved-label scorer.",
    )
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--app-source", type=Path, required=True)
    score.add_argument("--report", type=Path, required=True)
    score.add_argument("--baseline")
    score.add_argument("--bootstrap-samples", type=int, default=2000)
    score.add_argument("--seed", type=int, default=0)
    inspect_pool = annotation_commands.add_parser(
        "inspect", help="Verify package hashes and show work counts."
    )
    inspect_pool.add_argument("--pool", type=Path, required=True)

    sandbox = commands.add_parser(
        "sandbox", help="Create, list, and remove a project's copy by hand."
    )
    sandbox_sub = sandbox.add_subparsers(dest="sandbox_command")
    sandbox_create = sandbox_sub.add_parser("create", help="Copy a project.")
    _add_source_arguments(sandbox_create)
    sandbox_create.add_argument("--workspace", type=Path, required=True)
    sandbox_create.add_argument("--name", required=True)
    sandbox_remove = sandbox_sub.add_parser("remove", help="Remove a copy.")
    sandbox_remove.add_argument("--workspace", type=Path, required=True)
    sandbox_remove.add_argument("--name", required=True)
    sandbox_list = sandbox_sub.add_parser(
        "list", help="List the copies in a workspace."
    )
    sandbox_list.add_argument("--workspace", type=Path, required=True)

    return parser


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project", type=Path, required=True, help="The source project to measure."
    )
    _add_app_source_argument(parser)


def _add_app_source_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--app-source",
        type=Path,
        default=None,
        help=(
            "The research-rag tree to measure. Defaults to "
            "$RESEARCH_RAG_APP_SOURCE, then to the installed app."
        ),
    )


def _version() -> int:
    """Print this toolkit's version and the engine a run would measure.

    The engine is the one the environment names, then the installed app's own
    checkout. Saying "not found" with the flag that fixes it is more useful than a
    path that happens to be a directory, and the content digest is printed because a
    revision names committed work while a measurement reads the files on disk.
    """

    print(f"{PROGRAM} {toolkit_version()}")
    named = os.environ.get(APP_SOURCE_ENV, "").strip()
    try:
        engine = locate_engine(Path(named) if named else _engine_from_environment())
    except ExperimentError:
        print(
            f"engine: not found; pass --app-source, or set {APP_SOURCE_ENV} at a "
            "research-rag checkout"
        )
        return EXIT_VERIFIED
    described = engine.describe()
    revision = described.get("revision")
    state = "clean" if described.get("dirty") is False else "dirty"
    digest = str(described.get("content_sha256") or "")[:16]
    print(
        f"engine: {revision or 'no revision'} "
        f"({described.get('branch') or 'no branch'}, {state}) at {described['root']}, "
        f"content {digest or 'none'}"
    )
    return EXIT_VERIFIED


def _engine_from_environment() -> Path:
    from .engine.locate import PACKAGE_RELATIVE

    override = os.environ.get(APP_SOURCE_ENV)
    if override:
        return Path(override)
    # The installed package's own location, walked up to the tree that holds it,
    # which is the only place this harness can name an engine it was not given.
    import research_rag

    installed = Path(research_rag.__file__).resolve()
    for parent in installed.parents:
        if (parent / PACKAGE_RELATIVE).is_file():
            return parent
    return installed.parent


def _performance(args: argparse.Namespace) -> int:
    from .performance import run_performance

    result = run_performance(
        args.spec,
        args.app_source,
        args.workspace,
        args.runs,
        blocks=args.blocks,
        warmup_count=args.warmup_count,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return EXIT_VERIFIED


def _annotation(args: argparse.Namespace) -> int:
    from .annotation import build_pool, check_annotations, load_package

    if args.annotation_command == "score":
        from .annotation.scoring import score_handoff

        summary = score_handoff(
            args.input,
            args.app_source,
            args.report,
            baseline=args.baseline,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    elif args.annotation_command == "export":
        from .annotation.export import export_handoff

        summary = export_handoff(args.pool, args.judgments, args.output)
    elif args.annotation_command == "build":
        summary = build_pool(args.spec, args.output)
    elif args.annotation_command == "inspect":
        loaded = load_package(args.pool)
        pool = loaded["pool"]
        summary = {
            "pool_id": pool["pool_id"],
            "role": pool["role"],
            "questions": len(pool["questions"]),
            "items": len(pool["items"]),
            "pairs": len(pool["pairs"]),
            "integrity": "verified",
            "policy_ready": False,
        }
    else:
        summary = check_annotations(args.pool, args.judgments)
        if not args.details:
            summary = {
                **summary,
                **{
                    section: {
                        key: len(value) if isinstance(value, list) else value
                        for key, value in summary.get(section, {}).items()
                    }
                    for section in ("pending", "uncertain")
                },
            }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if (
        args.annotation_command == "check"
        and args.require_complete
        and summary["status"] != "complete"
    ):
        return EXIT_REFUSED
    return EXIT_VERIFIED


def _run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    engine = locate_engine(args.app_source or _engine_from_environment())
    if args.dry_run:
        return _dry_run(args, spec, engine)
    outcome = run_experiment(
        spec,
        app_source=engine,
        workspace=args.workspace,
        runs_directory=args.runs,
        keep_sandboxes=args.keep_sandboxes,
        validate_first=args.validate_first,
    )
    _report(outcome)
    print(outcome.comparison)
    print(f"record: {outcome.record_path}")
    return _status_for(outcome.verdict)


def _dry_run(args: argparse.Namespace, spec: RunSpec, engine: EngineSource) -> int:
    """Resolve every arm, measure nothing, and record what it proved.

    A dry run is a run of kind `preparation`: the same guard over the same source
    project, the same copies, the same settings resolution, and the same record. It
    runs no prepare command and no search, and its verdict is `prepared`, which says
    every arm could be built and the corpus was not written to — and says nothing at
    all about quality, because nothing was searched.

    It gets its own run identifier, so the run that measures after it collides with
    nothing and this one can be read on its own.
    """

    outcome = prepare_experiment(
        spec,
        app_source=engine,
        workspace=args.workspace,
        runs_directory=args.runs,
        keep_sandboxes=args.keep_sandboxes,
    )
    _report(outcome)
    for arm in outcome.record.get("arms") or []:
        settings = arm.get("settings") or {}
        sandbox = arm.get("sandbox") or {}
        print(
            f"arm {(arm.get('arm') or {}).get('name')}: "
            f"{(arm.get('arm') or {}).get('kind')}"
        )
        print(f"  sandbox:   {sandbox.get('root')}")
        checkout = arm.get("checkout")
        if checkout is not None:
            print(f"  checkout:  base {checkout.get('base_revision')}")
        print(f"  settings:  {str(settings.get('document_sha256'))[:16]}")
        print(f"  log:       {arm.get('log')}")
    print(
        f"Nothing was measured: this run prepared {outcome.record.get('prepared_arm_count')} of {len(spec.arms)} arms."
    )
    print(f"record: {outcome.record_path}")
    return _status_for(outcome.verdict)


def _status_for(verdict: str) -> int:
    if verdict in {VERDICT_VERIFIED, VERDICT_PREPARED}:
        # A preparation run succeeded; it is not a measurement, and its own status
        # says so without the exit code turning a clean preflight into a failure.
        return EXIT_VERIFIED
    if verdict == VERDICT_SOURCE_CHANGED:
        return EXIT_SOURCE_CHANGED
    if verdict == VERDICT_ENGINE_CHANGED:
        return EXIT_ENGINE_CHANGED
    if verdict in {VERDICT_INCOMPLETE, VERDICT_INTERRUPTED}:
        return EXIT_INCOMPLETE
    if verdict in {VERDICT_FAILED, VERDICT_NO_ARMS}:
        # A failed run and a run that measured nothing are both conditions a reader
        # acts on rather than measurements they can trust.
        return EXIT_REFUSED
    raise ExperimentError(
        f"This run's record carries the verdict {verdict!r}, which this command does "
        f"not know. A verdict nobody recognises is not a measurement, so it is "
        f"refused rather than reported as one."
    )


def _report(outcome: RunOutcome) -> None:
    """Say on stderr why a run is not a measurement, and nothing more.

    The verdict line is the whole claim. The reason is the first lines of the
    message the record carries, because a status alone leaves a reader guessing which
    arm refused, and the whole message is a command, its output, and a log path: it
    belongs in the record, and it is a page long on stderr.
    """

    verdict = outcome.verdict
    if verdict in {VERDICT_VERIFIED, VERDICT_PREPARED}:
        return
    print(
        f"{PROGRAM}: this run is not a verified measurement: {verdict}",
        file=sys.stderr,
    )
    for line in _first_lines(outcome.error):
        print(f"  {line}", file=sys.stderr)


#: How much of one line of a failure message reaches stderr, and how many lines.
#: An app message can name every setting it declares in a single line, and a
#: terminal is not the place for it: the record holds the whole message.
ERROR_LINE_CHARS = 160
ERROR_LINES = 4


def _first_lines(text: str | None, count: int = ERROR_LINES) -> list[str]:
    """The opening lines of a multi-line message, each cut to a readable length."""

    if not text:
        return []
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    chosen = lines[:count]
    if len(lines) > count:
        chosen.append(
            f"... and {len(lines) - count} more line(s); the record has the whole "
            f"message"
        )
    return [
        line if len(line) <= ERROR_LINE_CHARS else f"{line[:ERROR_LINE_CHARS]}..."
        for line in chosen
    ]


def _sandbox(args: argparse.Namespace) -> int:
    command = getattr(args, "sandbox_command", None)
    if command == "create":
        engine = locate_engine(args.app_source or _engine_from_environment())
        sandbox = create(
            args.workspace, name=args.name, source=args.project, generations=()
        )
        print(f"sandbox {sandbox.name}: {sandbox.root}")
        print(f"  generations: {', '.join(sandbox.generation_ids)}")
        print(f"  project id:  {sandbox.project_id}")
        print(f"  settings:    {sandbox.settings_file} (not written yet)")
        print(f"  engine:      {engine.root}")
        print(f"  record:      {sandbox.record_path}")
        print(
            "  A copy is addressed by path and is never registered: `research-rag "
            "init` would evict the original's record in the account's project "
            "registry."
        )
        return EXIT_VERIFIED
    if command == "remove":
        removed = remove(args.workspace, args.name)
        print(f"removed {removed}")
        return EXIT_VERIFIED
    if command == "list":
        found = list_sandboxes(args.workspace)
        if not found:
            print("no sandboxes in this workspace")
            return EXIT_VERIFIED
        for entry in found:
            print(
                f"{entry['name']}: {entry['root']}\n"
                f"  project id:  {entry.get('project_id')}\n"
                f"  generations: {', '.join(entry.get('generation_ids') or [])}"
            )
        return EXIT_VERIFIED
    print(
        "sandbox takes create, list, or remove: it makes and destroys a project's "
        "copy by hand, and every run makes its own.",
        file=sys.stderr,
    )
    return 1


def _verify(args: argparse.Namespace) -> int:
    before = snapshot(args.project)
    if args.expect is None:
        print(f"{args.project}: {before.digest()}")
        print(
            f"{before.path_count} paths, {before.byte_count} bytes, "
            f"{len(before.unreadable)} unreadable"
        )
        return EXIT_VERIFIED
    if before.digest() == args.expect:
        print(f"{args.project}: unchanged since the digest {args.expect}")
        return EXIT_VERIFIED
    print(
        f"{args.project}: is not the one with digest {args.expect}; it is now "
        f"{before.digest()}",
        file=sys.stderr,
    )
    return EXIT_SOURCE_CHANGED


def _compare(args: argparse.Namespace) -> int:
    record = read_record(args.run)
    if args.json:
        import json

        print(json.dumps(record, indent=2, ensure_ascii=False))
        return _status_for(str(record.get("verdict") or ""))
    print(
        render_comparison(
            record,
            modes=tuple(item.strip() for item in args.modes.split(",") if item.strip()),
            split=args.split,
        )
    )
    return _status_for(str(record.get("verdict") or ""))


def _inspect(args: argparse.Namespace) -> int:
    layout = read_layout(args.project)
    pointed = pointed_generation(layout)
    print(f"project:      {layout.root}")
    print(f"originals:    {layout.source_root}")
    print(f"review state: {layout.portable_root}")
    print(f"derived:      {layout.state_root}")
    if layout.relocated_runtime is not None:
        print(
            f"  the project relocated its own derived state to {layout.relocated_runtime}"
        )
    print(f"generation:   {pointed}")
    print(f"settings:     {layout.root / layout.settings_relative}")
    print(
        f"generations:  {', '.join(sorted(_names(layout.generations_root))) or 'none'}"
    )
    print("  guarded: every path in the project, including the generations, the")
    print("  review state, and any file at the root, except the process state a")
    print(f"  serving app rewrites on its own: {', '.join(volatile_paths())}")
    print("  a copy carries none of that, and creates the runtime's empty")
    print(f"  directories itself: {', '.join(layout.empty_runtime_directories)}")
    print("  any other change during a run, by the app or by a person, disqualifies it")
    print(
        f"A run copies the originals, the review state, and generation {pointed}, and "
        "writes every byte it produces inside the directories it was given. The "
        "account's project registry is never written: a copy is addressed by path."
    )
    return EXIT_VERIFIED


def _names(root: Path) -> list[str]:
    return [path.name for path in root.iterdir()] if root.is_dir() else []


__all__ = [
    "APP_SOURCE_ENV",
    "EXIT_ENGINE_CHANGED",
    "EXIT_INCOMPLETE",
    "EXIT_REFUSED",
    "EXIT_SOURCE_CHANGED",
    "EXIT_VERIFIED",
    "PROGRAM",
    "RECORD_NAME",
    "build_parser",
    "main",
]
