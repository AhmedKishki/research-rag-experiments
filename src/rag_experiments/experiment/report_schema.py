"""Whether the report an arm's harness wrote is the measurement that was asked for.

A report that exists is not a report that answers the question. Its schema version
may not be the one the engine's own harness declares, it may not carry a mode the
specification asked for, it may have been produced from a different judged set or a
different output depth, its reranked row may describe searches that quietly did not
rerank, and its per-query rows may be a different set of queries from the one this
run selected. Every one of those is a report whose numbers would be printed beside
the settings that produced them while describing neither.

So a report is read against a contract rather than sampled for keys, and every fact
the contract needs must be present: an absent identity is refused rather than read as
an absent disagreement, because a report that states no judged path and one that
states the wrong one are equally unusable and only one of them looks like a pass.

The contract comes from three places and nowhere else: the engine's own harness
(`engine/harness.py` reads its declared constants), the command this harness built,
and the judged file this run handed it, whose selection is computed here from the
same file and the same flags the harness was given.

Every fact compared comes from the engine under test's own harness, from the command
this harness built, or from the judged file. Nothing here guesses: a mismatch means
the engine and this run disagree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..engine.harness import HarnessContract
from ..errors import ExperimentError

#: The keys one per-query run must carry for a depth claim to mean anything. The
#: harness reports its depth, and a row that reports none cannot be checked against
#: the depth the specification asked for.
RUN_DEPTH_KEY = "top_k"

#: The identity a per-query run must carry for a count to mean anything. A row with
#: no query id cannot be matched against the selection this run asked for.
RUN_IDENTITY_KEY = "query_id"

#: The evidence keys the harness's reranked row carries, read from the engine's
#: search payload: whether the reranker ran, whether it was asked to, and whether it
#: fell back instead.
RERANKED_KEY = "reranked"
RERANK_REQUESTED_KEY = "rerank_requested"
RERANK_FALLBACK_KEY = "rerank_fallback"

#: How many candidates a reranked row had to rank. A row with none has nothing to
#: rerank, so it cannot evidence that the reranker ran, and a report that counts it
#: as applied is counting a query that never reached one.
CANDIDATE_COUNT_KEY = "candidate_count"

#: The judged-set fields this harness reads. These are the app's judged-set schema,
#: version 1, as its own harness reads the same file: each query names one target and
#: carries a class, and each target is named. Nothing else in the file is used.
JUDGED_SCHEMA_VERSION = 1
JUDGED_QUERY_ID = "query_id"
JUDGED_CLASS = "class"
JUDGED_TARGET_ID = "target_id"

#: The harness flags that select which queries are measured. A run that states none
#: of them is measured over every query of every class the file holds, which is what
#: the engine does when it is given no selection.
SELECTION_FLAGS = ("classes", "limit", "skip_targets")

#: Where a per-mode count is stated in the summary. The harness aggregates each mode
#: into an `overall` block carrying the number of rows it summarised.
SUMMARY_COUNT = "overall"


class ReportValidationError(ExperimentError):
    """A report that exists and does not answer the question that was asked.

    A separate type from `ExperimentError` because the condition is a report, not a
    refusal by the harness: the arm ran, exited cleanly, and produced something
    unusable, and a reader has to be told which part of it is wrong.
    """


@dataclass(frozen=True, slots=True)
class QuerySelection:
    """The queries a run asked to be measured, computed from the judged file.

    The judged file is the same file the harness read, and the flags are the same
    flags it was given, so this is the query set the report's rows are checked
    against rather than a set this harness chose.
    """

    path: Path
    query_ids: tuple[str, ...]
    total_query_count: int
    classes: tuple[str, ...]
    limit: int | None
    skipped_targets: tuple[str, ...]
    explicit: bool

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "query_count": len(self.query_ids),
            "total_query_count": self.total_query_count,
            "query_ids": list(self.query_ids),
            "classes": list(self.classes),
            "limit": self.limit,
            "skipped_targets": list(self.skipped_targets),
            "explicit_selection": self.explicit,
        }


@dataclass(frozen=True, slots=True)
class ModeFacts:
    """What one mode's rows say, as read back from the report."""

    label: str
    query_ids: tuple[str, ...]
    duplicates: tuple[str, ...] = ()
    unknown_query_ids: tuple[str, ...] = ()
    missing_query_ids: tuple[str, ...] = ()
    wrong_depths: tuple[str, ...] = ()
    deep_rows_without_depth: tuple[str, ...] = ()
    rerank_not_applied: tuple[str, ...] = ()
    rerank_fallbacks: tuple[str, ...] = ()
    rerank_without_candidates: tuple[str, ...] = ()
    summary_count: int | None = None


@dataclass(frozen=True, slots=True)
class ReportFacts:
    """What one report says about itself, as read back from its own bytes."""

    path: Path
    sha256: str
    schema_version: int
    modes: tuple[str, ...]
    top_k: int | None
    deep_top_k: int | None
    judged_path: str
    judged_sha256: str | None
    judged_query_count: int | None
    evaluated_query_count: int | None
    selected_classes: tuple[str, ...]
    skipped_targets: tuple[str, ...]
    run_count: int
    deep_run_count: int
    by_mode: dict[str, ModeFacts] = field(default_factory=dict)
    deep_top_ks: tuple[int, ...] = ()
    deep_rows_without_depth: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "modes": list(self.modes),
            "top_k": self.top_k,
            "deep_top_k": self.deep_top_k,
            "judged_path": self.judged_path,
            "judged_sha256": self.judged_sha256,
            "judged_query_count": self.judged_query_count,
            "evaluated_query_count": self.evaluated_query_count,
            "selected_classes": list(self.selected_classes),
            "skipped_targets": list(self.skipped_targets),
            "run_count": self.run_count,
            "deep_run_count": self.deep_run_count,
            "by_mode": {
                label: {
                    "query_ids": list(facts.query_ids),
                    "summary_count": facts.summary_count,
                    "duplicates": list(facts.duplicates),
                    "missing_query_ids": list(facts.missing_query_ids),
                    "wrong_depths": list(facts.wrong_depths),
                    "rerank_not_applied": list(facts.rerank_not_applied),
                    "rerank_fallbacks": list(facts.rerank_fallbacks),
                    "rerank_without_candidates": list(facts.rerank_without_candidates),
                }
                for label, facts in self.by_mode.items()
            },
        }


def select_queries(
    judged_path: Path,
    *,
    classes: Any = None,
    limit: Any = None,
    skip_targets: Any = None,
) -> QuerySelection:
    """Compute the queries a run measures from the judged file and its own flags.

    The order of the steps is the engine's: keep the queries whose class was asked
    for and whose target was not skipped, then keep the first `limit` of those. The
    expected query set is used to check that a report's rows are the rows this run
    asked for, so it has to be the engine's set and not a second implementation of a
    similar one: a different order or a different reading of a class would refuse a
    report that measured exactly what was asked.

    A judged file with a repeated query id is refused here rather than compared: two
    rows would then share one identity and neither could be matched.
    """

    document = _read_judged(judged_path)
    queries = document["queries"]
    targets = document["targets"]
    known_targets = {str(target[JUDGED_TARGET_ID]) for target in targets}
    present = {str(query[JUDGED_CLASS]) for query in queries}

    explicit = False
    wanted_classes = _names(classes, "classes")
    if wanted_classes:
        explicit = True
        unknown = sorted(set(wanted_classes) - present)
        if unknown:
            raise ExperimentError(
                f"The judged set at {judged_path} holds no query of class "
                f"{', '.join(unknown)}; it holds {', '.join(sorted(present))}. "
                "Refusing to measure a class this file does not have."
            )
    else:
        wanted_classes = tuple(sorted(present))

    skipped = tuple(sorted(_names(skip_targets, "skip_targets")))
    if skipped:
        explicit = True
        unknown_targets = sorted(set(skipped) - known_targets)
        if unknown_targets:
            raise ExperimentError(
                f"The judged set at {judged_path} names no target "
                f"{', '.join(unknown_targets)} to skip."
            )

    kept = [
        str(query[JUDGED_QUERY_ID])
        for query in queries
        if str(query[JUDGED_CLASS]) in wanted_classes
        and str(query[JUDGED_TARGET_ID]) not in skipped
    ]
    count = _limit(limit)
    if count is not None:
        explicit = True
        kept = kept[:count]
    if not kept:
        raise ExperimentError(
            f"The judged set at {judged_path} selects no query: classes "
            f"{', '.join(wanted_classes)}, skipped targets "
            f"{', '.join(skipped) or 'none'}, limit {count if count else 'none'}. "
            "A run measures nothing from a split with no queries."
        )
    repeated = tuple(sorted({name for name in kept if kept.count(name) > 1}))
    if repeated:
        raise ExperimentError(
            f"The judged set at {judged_path} repeats the query id "
            f"{', '.join(repeated)}. A query is one row per mode, so a repeated id "
            "cannot be matched against the selection."
        )
    return QuerySelection(
        path=judged_path,
        query_ids=tuple(kept),
        total_query_count=len(queries),
        classes=wanted_classes,
        limit=count,
        skipped_targets=skipped,
        explicit=explicit,
    )


def validate_report(
    path: Path,
    *,
    contract: HarnessContract,
    requested_modes: tuple[str, ...],
    selection: QuerySelection,
    judged_sha256: str,
    top_k: int | None,
    deep_top_k: int | None,
    arm: str,
    split: str,
) -> ReportFacts:
    """Read one arm's report and refuse it unless it is the measurement asked for.

    `requested_modes`, `top_k`, and `deep_top_k` are the values this run put on the
    command line, and `selection` is the query set the judged file yields under those
    same flags. They are compared with what the report says it did, so a report
    produced from different arguments, a different judged file, or a different
    selection cannot be printed as though it described this arm.
    """

    text = _read(path, arm=arm, split=split)
    facts = _facts(path, text, requested_top_k=top_k)
    _require_schema(facts, contract, arm=arm, split=split)
    _require_modes(facts, contract, requested_modes, arm=arm, split=split)
    _require_judged(facts, selection, judged_sha256, arm=arm, split=split)
    _require_depths(facts, top_k, deep_top_k, arm=arm, split=split)
    _require_queries(facts, contract, requested_modes, selection, arm=arm, split=split)
    _require_rerank(facts, contract, requested_modes, arm=arm, split=split)
    return facts


def _read(path: Path, *, arm: str, split: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote no report this harness can read at "
            f"{path}: {exc}"
        ) from exc


def _facts(path: Path, text: str, *, requested_top_k: int | None = None) -> ReportFacts:
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReportValidationError(
            f"The report arm {path.name!r} wrote is not readable JSON: {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ReportValidationError(
            f"The report at {path} is not an object; this harness reads the app's "
            "own report shape and cannot interpret another."
        )
    settings = document.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    judgments = document.get("judgments")
    judgments = judgments if isinstance(judgments, dict) else {}
    summary = document.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    runs = [run for run in (document.get("runs") or []) if isinstance(run, dict)]
    deep_runs = [
        run for run in (document.get("deep_runs") or []) if isinstance(run, dict)
    ]
    return ReportFacts(
        path=path,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        schema_version=_plain_int(document.get("schema_version"), default=-1),
        modes=tuple(str(mode) for mode in summary),
        # A depth of zero is a stated decision, not an absent one: the harness writes
        # zero for a disabled deep pass, and reading that as "no depth recorded"
        # would refuse a report that said exactly what it did.
        top_k=_nonnegative_int(settings.get("top_k")),
        deep_top_k=_nonnegative_int(settings.get("deep_top_k")),
        judged_path=str(judgments.get("path") or ""),
        judged_sha256=_optional_text(judgments.get("sha256")),
        judged_query_count=_count(judgments.get("query_count")),
        evaluated_query_count=_count(judgments.get("evaluated_query_count")),
        selected_classes=tuple(
            str(item) for item in (settings.get("selected_classes") or [])
        ),
        skipped_targets=tuple(
            str(item) for item in (settings.get("skipped_targets") or [])
        ),
        run_count=len(runs),
        deep_run_count=len(deep_runs),
        deep_top_ks=tuple(
            value
            for value in (_nonnegative_int(run.get(RUN_DEPTH_KEY)) for run in deep_runs)
            if value is not None
        ),
        by_mode=_by_mode(runs, summary, requested_top_k),
        deep_rows_without_depth=tuple(
            str(run.get(RUN_IDENTITY_KEY) or "")
            for run in deep_runs
            if _nonnegative_int(run.get(RUN_DEPTH_KEY)) is None
        ),
    )


def _by_mode(
    runs: list[dict[str, Any]], summary: dict[str, Any], requested_top_k: int | None
) -> dict[str, ModeFacts]:
    """Group the rows by the label they carry, and read what each group says.

    The grouping is by the row's own label rather than by the mode this run asked
    for, because a report whose rows carry a label this run never asked for is a
    report that measured something else, and saying so is the point.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run.get("mode") or ""), []).append(run)
    facts: dict[str, ModeFacts] = {}
    for label, rows in grouped.items():
        identifiers = tuple(str(row.get(RUN_IDENTITY_KEY) or "") for row in rows)
        counts = [identifier for identifier in identifiers if identifier]
        repeated = tuple(sorted({name for name in counts if counts.count(name) > 1}))
        summary_block = summary.get(label)
        summary_block = summary_block if isinstance(summary_block, dict) else {}
        overall = summary_block.get(SUMMARY_COUNT)
        overall = overall if isinstance(overall, dict) else {}
        facts[label] = ModeFacts(
            label=label,
            query_ids=identifiers,
            duplicates=repeated,
            wrong_depths=_wrong_depths(rows, requested_top_k),
            rerank_not_applied=tuple(
                str(row.get(RUN_IDENTITY_KEY) or "")
                for row in rows
                if row.get(RERANKED_KEY) is not True
            ),
            rerank_fallbacks=tuple(
                str(row.get(RUN_IDENTITY_KEY) or "")
                for row in rows
                if row.get(RERANK_FALLBACK_KEY)
            ),
            rerank_without_candidates=tuple(
                str(row.get(RUN_IDENTITY_KEY) or "")
                for row in rows
                if row.get(RERANKED_KEY) is True
                and _nonnegative_int(row.get(CANDIDATE_COUNT_KEY)) == 0
            ),
            summary_count=_count(overall.get("query_count")),
        )
    return facts


def _wrong_depths(rows: list[dict[str, Any]], requested: int | None) -> tuple[str, ...]:
    """Rows whose depth is absent, or is not the depth this run asked for.

    A stated depth that differs is the more serious of the two: the summary would be
    a mean over rows measured at more than one depth while stating one, and a reader
    has no way to see that from the row itself.
    """

    wrong: list[str] = []
    for row in rows:
        stated = _nonnegative_int(row.get(RUN_DEPTH_KEY))
        if stated is None or (requested is not None and stated != requested):
            wrong.append(str(row.get(RUN_IDENTITY_KEY) or ""))
    return tuple(wrong)


def _require_schema(
    facts: ReportFacts, contract: HarnessContract, *, arm: str, split: str
) -> None:
    if facts.schema_version != contract.report_schema_version:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote a report of schema "
            f"{facts.schema_version}, and {contract.harness} declares "
            f"{contract.report_schema_version}. The two are different measurements, "
            f"so the numbers are not printed beside the settings that produced "
            f"them. Re-run with the engine under test, or point --app-source at the "
            f"tree whose harness writes schema {facts.schema_version}."
        )


def _require_modes(
    facts: ReportFacts,
    contract: HarnessContract,
    requested: tuple[str, ...],
    *,
    arm: str,
    split: str,
) -> None:
    if facts.run_count == 0:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote a report with no per-query runs at "
            f"{facts.path}. It measured nothing."
        )
    missing = [
        mode
        for mode in requested
        if not any(contract.matches(mode, key) for key in facts.modes)
    ]
    if missing:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} asked for {', '.join(missing)}, and its "
            f"report at {facts.path} summarises {', '.join(facts.modes) or 'nothing'}. "
            f"A mode that was not measured has no number, and printing its column "
            f"as a dash would read as a mode that found nothing."
        )


def _require_judged(
    facts: ReportFacts,
    selection: QuerySelection,
    judged_sha256: str,
    *,
    arm: str,
    split: str,
) -> None:
    """The report must name the file this run handed it, and the queries it ran.

    An absent path is refused rather than skipped. A report that names no judged
    file and one that names the wrong file are equally unusable, and only the second
    reads as a disagreement a reader can act on.

    The harness records the path it read but not a digest of it, so the digest is
    compared when the report carries one. In both cases the run keeps its own copy of
    the bytes and its digest beside the report, which is what lets a reader confirm
    the two files were the same one.
    """

    if not facts.judged_path:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote a report naming no judged file at "
            f"{facts.path}. This run measured {selection.path}, and a report that "
            f"does not say which file it read cannot be compared with it."
        )
    if facts.judged_path != str(selection.path):
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote a report measured from "
            f"{facts.judged_path}, and this run measured {selection.path}. Two judged "
            f"sets in one table are two different questions."
        )
    if facts.judged_sha256 and facts.judged_sha256 != judged_sha256:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote a report whose digest for "
            f"{selection.path} is {facts.judged_sha256[:16]}, and this run measured "
            f"bytes digesting to {judged_sha256[:16]}. The file changed under the "
            f"run, so the number describes neither version."
        )
    if facts.judged_query_count is None:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote a report stating no query count for "
            f"{selection.path} at {facts.path}. The count is what says the whole file "
            f"was read rather than part of it."
        )
    if facts.judged_query_count != selection.total_query_count:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote a report reading "
            f"{facts.judged_query_count} queries from {selection.path}, and that file "
            f"holds {selection.total_query_count}. A different judged set measured "
            f"under this split's name."
        )
    if facts.evaluated_query_count is None:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote a report stating no evaluated query "
            f"count for {selection.path}. Without it, a report that measured one "
            f"query out of thirty reads the same as one that measured all thirty."
        )
    if facts.evaluated_query_count == 0:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} evaluated no query; {facts.path} reports an "
            f"evaluated query count of zero, and this run selected "
            f"{len(selection.query_ids)} of {selection.path}."
        )
    if facts.evaluated_query_count != len(selection.query_ids):
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} evaluated "
            f"{facts.evaluated_query_count} queries of {selection.path}, and this run "
            f"selected {len(selection.query_ids)}"
            f"{_selection_note(selection)}. A report measured over a different "
            f"selection is a different measurement of the same file."
        )
    # When this run named a selection, the report has to name the same one. A class
    # or a skipped target that changed nothing measurable in this file would
    # otherwise pass on the query set alone while the two runs disagree about what
    # they selected.
    if selection.explicit and set(facts.selected_classes) != set(selection.classes):
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} measured classes "
            f"{', '.join(facts.selected_classes) or 'none stated'} and this run asked "
            f"for {', '.join(selection.classes)} of {selection.path}. Two runs "
            f"selecting different classes are two different query sets, whichever "
            f"file they were read from."
        )
    if selection.explicit and set(facts.skipped_targets) != set(
        selection.skipped_targets
    ):
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} skipped targets "
            f"{', '.join(facts.skipped_targets) or 'none'} and this run skipped "
            f"{', '.join(selection.skipped_targets) or 'none'} of "
            f"{selection.path}."
        )


def _selection_note(selection: QuerySelection) -> str:
    """How the queries were chosen, in one clause a reader can check against."""

    parts = [f"classes {', '.join(selection.classes)}"]
    if selection.skipped_targets:
        parts.append(f"skipping {', '.join(selection.skipped_targets)}")
    if selection.limit is not None:
        parts.append(f"limited to {selection.limit}")
    return " (" + ", ".join(parts) + ")"


def _require_depths(
    facts: ReportFacts,
    top_k: int | None,
    deep_top_k: int | None,
    *,
    arm: str,
    split: str,
) -> None:
    """Every depth stated must be the one this run asked for, and must be stated.

    An absent depth is refused. `deep_top_k` of zero is the engine's way of saying
    the deep pass was not run, and it is a stated decision, so it is compared like
    any other value rather than read as nothing recorded.
    """

    if facts.top_k is None:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote a report stating no top_k at "
            f"{facts.path}. A success@k over an unstated depth is not a number any "
            f"reader can compare with another run's."
        )
    if top_k is not None and facts.top_k != top_k:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} was asked for --top-k {top_k} and its "
            f"report at {facts.path} records top_k={facts.top_k}. A success@k over a "
            f"different depth is a different number."
        )
    if deep_top_k is None:
        return
    if facts.deep_top_k is None:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} was asked for a deep pass at "
            f"--deep-top-k {deep_top_k}, and its report at {facts.path} states no "
            f"deep depth at all."
        )
    if facts.deep_top_k != deep_top_k:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} was asked for --deep-top-k {deep_top_k} "
            f"and its report at {facts.path} records deep_top_k={facts.deep_top_k}."
        )
    if facts.deep_rows_without_depth:
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} wrote deep rows stating no "
            f"{RUN_DEPTH_KEY}: {', '.join(facts.deep_rows_without_depth[:5])} in "
            f"{facts.path}. A deep row measured at an unstated depth cannot be "
            f"compared with one measured at {deep_top_k}."
        )
    if facts.deep_run_count and any(depth != deep_top_k for depth in facts.deep_top_ks):
        raise ReportValidationError(
            f"Arm {arm!r} split {split!r} recorded a deep depth of "
            f"{sorted(set(facts.deep_top_ks))} beside a stated deep_top_k of "
            f"{deep_top_k} in {facts.path}. The rows and the stated depth disagree."
        )


def _require_queries(
    facts: ReportFacts,
    contract: HarnessContract,
    requested: tuple[str, ...],
    selection: QuerySelection,
    *,
    arm: str,
    split: str,
) -> None:
    """The rows must be this run's rows: one per query per mode, no more and fewer.

    A mode whose rows are a different set of queries is a measurement of something
    else, and a summary count that disagrees with the rows beneath it is a table a
    reader would read at face value.
    """

    for label, mode_facts in facts.by_mode.items():
        asked = next(
            (mode for mode in requested if contract.matches(mode, label)), None
        )
        if asked is None:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} wrote rows labelled {label!r} in "
                f"{facts.path}, which is not one of the modes this run asked for "
                f"({', '.join(requested)}). A row for a mode nobody asked about is a "
                f"row the comparison would not show, and its numbers would still be "
                f"in the report."
            )
        if mode_facts.duplicates:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} wrote "
                f"{len(mode_facts.duplicates)} rows for the same query in mode "
                f"{label!r}: {', '.join(mode_facts.duplicates)}. One query is one row "
                f"per mode, so a repeated query id is a repeated measurement."
            )
        blank = [name for name in mode_facts.query_ids if not name]
        if blank:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} wrote {len(blank)} rows in mode "
                f"{label!r} naming no query, in {facts.path}. A row with no identity "
                f"cannot be matched against the selection."
            )
        if mode_facts.wrong_depths:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} wrote rows in mode {label!r} stating no "
                f"{RUN_DEPTH_KEY}: {', '.join(mode_facts.wrong_depths)}."
            )
        expected = set(selection.query_ids)
        present = set(mode_facts.query_ids)
        unknown = sorted(present - expected)
        if unknown:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} measured queries "
                f"{', '.join(unknown[:5])} in mode {label!r}, which this run did not "
                f"select from {selection.path}"
                f"{_selection_note(selection)}."
            )
        missing = sorted(expected - present)
        if missing:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} measured {len(present)} of the "
                f"{len(expected)} selected queries in mode {label!r}; "
                f"{', '.join(missing[:5])} are missing from {facts.path}. A mode "
                f"missing a query is a mean over a different query set than the one "
                f"the arm's siblings were measured over."
            )
        if mode_facts.summary_count is None:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} wrote no summary count for mode "
                f"{label!r} in {facts.path}, so the row a reader reads cannot be "
                f"checked against the rows beneath it."
            )
        if mode_facts.summary_count != len(mode_facts.query_ids):
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} summarises "
                f"{mode_facts.summary_count} queries for mode {label!r} and wrote "
                f"{len(mode_facts.query_ids)} rows for it in {facts.path}. The mean a "
                f"reader reads is over a different number of queries than the rows "
                f"below it."
            )


def _require_rerank(
    facts: ReportFacts,
    contract: HarnessContract,
    requested: tuple[str, ...],
    *,
    arm: str,
    split: str,
) -> None:
    """Every reranked row must be reranked, or the arm is refused.

    The engine reports a reranker that was unavailable as a fallback: the search
    runs, the row exists, and the number is a plain hybrid ranking wearing a
    reranked label. It also reports a query that returned no candidates, which never
    reached a reranker at all: counting that row as applied would claim reranking
    from a search that had nothing to rank. Both are refused, and the refusal names
    the queries so a reader can see which rows are at fault rather than only that
    some were.
    """

    asked = contract.rerank_modes(requested)
    if not asked:
        return
    for label, mode_facts in facts.by_mode.items():
        if not contract.rerank_modes((label,)):
            continue
        if mode_facts.rerank_fallbacks:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} reports "
                f"{len(mode_facts.rerank_fallbacks)} of {len(mode_facts.query_ids)} "
                f"queries in mode {label!r} whose reranker fell back "
                f"({', '.join(mode_facts.rerank_fallbacks[:5])}), so that row is not a "
                f"reranked measurement. The report is at {facts.path}."
            )
        if mode_facts.rerank_not_applied:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} reports {RERANKED_KEY!r} other than "
                f"true for {len(mode_facts.rerank_not_applied)} of "
                f"{len(mode_facts.query_ids)} queries in mode {label!r} "
                f"({', '.join(mode_facts.rerank_not_applied[:5])}) in {facts.path}. "
                f"A reranked row where some queries were not reranked is a mean over "
                f"two different kinds of search."
            )
        if mode_facts.rerank_without_candidates:
            raise ReportValidationError(
                f"Arm {arm!r} split {split!r} counts "
                f"{len(mode_facts.rerank_without_candidates)} of "
                f"{len(mode_facts.query_ids)} queries in mode {label!r} as reranked "
                f"although they returned no candidates "
                f"({', '.join(mode_facts.rerank_without_candidates[:5])}). A query "
                f"with nothing to rank never reached the reranker, so it cannot "
                f"evidence that one ran; the condition is declared at "
                f"{CANDIDATE_COUNT_KEY!r} in {facts.path} and the row must say "
                f"which queries it does not claim."
            )


def _read_judged(path: Path) -> dict[str, Any]:
    """Read the judged file's own identity fields, refusing anything unreadable.

    These are the app's judged-set schema as its harness reads it: a version, a list
    of targets each named, and a list of queries each naming one target, a class,
    and its own id. A file that does not carry them cannot be measured, and the
    harness refuses such a file before it measures anything; this says so first, with
    the field that is missing.
    """

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read the judged set: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ExperimentError(f"The judged set is not an object: {path}")
    version = document.get("schema_version")
    if version != JUDGED_SCHEMA_VERSION:
        raise ExperimentError(
            f"{path} declares judged-set schema_version {version!r}; this harness "
            f"reads {JUDGED_SCHEMA_VERSION}."
        )
    queries = document.get("queries")
    targets = document.get("targets")
    if not isinstance(queries, list) or not isinstance(targets, list):
        raise ExperimentError(f"The judged set has no `queries` or `targets`: {path}")
    if not targets:
        raise ExperimentError(f"The judged set has no targets: {path}")
    for index, target in enumerate(targets, 1):
        if (
            not isinstance(target, dict)
            or not str(target.get(JUDGED_TARGET_ID) or "").strip()
        ):
            raise ExperimentError(
                f"Target {index} of {path} names no {JUDGED_TARGET_ID!r}, so a query "
                f"cannot be matched to it."
            )
    for index, query in enumerate(queries, 1):
        if not isinstance(query, dict):
            raise ExperimentError(f"Query {index} of {path} is not an object.")
        for key in (JUDGED_QUERY_ID, JUDGED_CLASS, JUDGED_TARGET_ID):
            if not str(query.get(key) or "").strip():
                raise ExperimentError(
                    f"Query {index} of {path} carries no {key!r}, so it has no "
                    f"identity this harness can match a report row against."
                )
    if not queries:
        raise ExperimentError(f"The judged set has no queries: {path}")
    return {"queries": queries, "targets": targets}


def _names(value: Any, flag: str) -> tuple[str, ...]:
    """Read one selection flag as the engine reads it: a comma-separated list."""

    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ExperimentError(
        f"The harness entry {flag!r} takes a comma-separated list or a list of "
        f"names, not {value!r}."
    )


def _limit(value: Any) -> int | None:
    """Read a limit as the engine reads it: falsy means no limit at all."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ExperimentError(f"The harness entry 'limit' is not a number: {value!r}")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(
            f"The harness entry 'limit' is not a number: {value!r}"
        ) from exc
    return count if count > 0 else None


def _optional_text(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _nonnegative_int(value: Any) -> int | None:
    """A stated whole number, including zero, which is a decision and not an absence."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _plain_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _count(value: Any) -> int | None:
    """A count the report states, including zero, which is itself a refusal."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


__all__ = [
    "CANDIDATE_COUNT_KEY",
    "JUDGED_CLASS",
    "JUDGED_QUERY_ID",
    "JUDGED_SCHEMA_VERSION",
    "JUDGED_TARGET_ID",
    "RERANKED_KEY",
    "RERANK_FALLBACK_KEY",
    "RERANK_REQUESTED_KEY",
    "RUN_DEPTH_KEY",
    "RUN_IDENTITY_KEY",
    "SELECTION_FLAGS",
    "ModeFacts",
    "QuerySelection",
    "ReportFacts",
    "ReportValidationError",
    "select_queries",
    "validate_report",
]
