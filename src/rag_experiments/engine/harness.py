"""What the engine's own evaluation harness declares, read out of its source.

The harness is a script, so it is not imported to be asked: importing it would run
its imports, which pull in the whole retrieval stack. Its module-level constants
are read from its syntax tree instead, which is the same rule the rest of this
toolkit follows — a name the app owns is read from the app, never written here.

Three facts come from it and all three decide whether a report can be believed:
the modes it measures, the label its reranked mode carries, and the report schema
version it writes. A harness whose report shape moved is refused rather than
parsed into columns that would print as a dash.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from ..errors import ExperimentError
from .locate import EngineSource

#: The constants this harness reads, and what each one is. A constant the script
#: no longer declares is a refusal rather than a default, because a default here
#: would be this toolkit's guess about what the engine measured.
MODE_CONSTANT = "MODES"
RERANK_CONSTANT = "RERANK_MODE"
SCHEMA_CONSTANT = "REPORT_SCHEMA_VERSION"

#: How large a harness script may be before it is refused rather than parsed. The
#: script under test is a source file, not an archive.
MAXIMUM_HARNESS_BYTES = 4 << 20


@dataclass(frozen=True, slots=True)
class HarnessContract:
    """What one engine's harness says it measures and what it writes."""

    harness: Path
    modes: tuple[str, ...]
    rerank_mode: str
    report_schema_version: int

    def rerank_modes(self, requested: tuple[str, ...]) -> tuple[str, ...]:
        """Which of `requested` are reranked rows.

        The harness labels one row per reranker, so a run that names two models has
        two labels for one mode. Both are reranked rows, and both are matched.
        """

        return tuple(
            mode
            for mode in requested
            if mode == self.rerank_mode or mode.startswith(f"{self.rerank_mode}[")
        )

    def matches(self, requested: str, summary_key: str) -> bool:
        """Whether one summary key answers for one requested mode."""

        return summary_key == requested or (
            requested == self.rerank_mode
            and summary_key.startswith(f"{self.rerank_mode}[")
        )

    def describe(self) -> dict[str, object]:
        return {
            "harness": str(self.harness),
            "modes": list(self.modes),
            "rerank_mode": self.rerank_mode,
            "report_schema_version": self.report_schema_version,
        }


def harness_contract(engine: EngineSource) -> HarnessContract:
    """Read the modes, the rerank label, and the report version from the harness."""

    script = engine.harness
    if not script.is_file():
        raise ExperimentError(
            f"{engine.root} ships no evaluation harness at {script}, so there is "
            "nothing to say which modes it measures."
        )
    size = script.stat().st_size
    if size > MAXIMUM_HARNESS_BYTES:
        raise ExperimentError(
            f"{script} is {size} bytes, larger than this harness will parse "
            f"({MAXIMUM_HARNESS_BYTES} bytes)."
        )
    constants = _module_constants(script)
    missing = [
        name
        for name in (MODE_CONSTANT, RERANK_CONSTANT, SCHEMA_CONSTANT)
        if name not in constants
    ]
    if missing:
        raise ExperimentError(
            f"{script} declares no {', '.join(missing)} at module level, so this "
            "harness cannot say which modes it measures or which report shape it "
            "writes. Point --app-source at a research-rag checkout whose harness "
            "declares them."
        )
    modes = constants[MODE_CONSTANT]
    rerank = constants[RERANK_CONSTANT]
    version = constants[SCHEMA_CONSTANT]
    if not isinstance(modes, tuple) or not all(isinstance(mode, str) for mode in modes):
        raise ExperimentError(f"{script} declares {MODE_CONSTANT} as {modes!r}")
    if not isinstance(rerank, str) or rerank not in modes:
        raise ExperimentError(
            f"{script} declares {RERANK_CONSTANT} as {rerank!r}, which is not one of "
            f"its {MODE_CONSTANT}"
        )
    if not isinstance(version, int) or isinstance(version, bool):
        raise ExperimentError(
            f"{script} declares {SCHEMA_CONSTANT} as {version!r}, which is not a "
            "report version"
        )
    return HarnessContract(
        harness=script,
        modes=tuple(modes),
        rerank_mode=rerank,
        report_schema_version=version,
    )


class _Unevaluable(Exception):
    """A module-level constant this harness declines to evaluate."""


def _module_constants(script: Path) -> dict[str, object]:
    """The module-level constants of a script, resolved against each other.

    A harness declares its modes from its own rerank label rather than repeating
    the string, so a constant is evaluated against the constants already read: the
    expression is compiled and run with no builtins and a namespace holding nothing
    but the other constants. Anything that needs a function, an import, or a name
    this harness did not read is reported as absent, because evaluating it here would
    mean reimplementing code the engine owns.
    """

    try:
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise ExperimentError(f"Cannot read the harness at {script}: {exc}") from exc
    pending: list[tuple[str, ast.expr]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            pending.append((target.id, node.value))

    values: dict[str, object] = {}
    while pending:
        remaining: list[tuple[str, ast.expr]] = []
        progressed = False
        for name, expression in pending:
            try:
                values[name] = _evaluate(expression, values)
            except _Unevaluable:
                remaining.append((name, expression))
            else:
                progressed = True
        if not progressed:
            break
        pending = remaining
    return values


def _evaluate(expression: ast.expr, names: dict[str, object]) -> object:
    try:
        return ast.literal_eval(expression)
    except (ValueError, SyntaxError, TypeError):
        pass
    namespace = {
        name: value
        for name, value in names.items()
        if isinstance(value, (str, bool, int, float, tuple, list, frozenset))
    }
    try:
        return eval(
            compile(ast.Expression(expression), "<harness-constant>", "eval"),
            {"__builtins__": {}},
            namespace,
        )
    except Exception as exc:
        raise _Unevaluable(str(exc)) from exc


__all__ = [
    "MODE_CONSTANT",
    "RERANK_CONSTANT",
    "SCHEMA_CONSTANT",
    "HarnessContract",
    "harness_contract",
]
