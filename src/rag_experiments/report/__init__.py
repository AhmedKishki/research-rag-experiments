"""The record a run leaves, and the table that reads it."""

from __future__ import annotations

from .compare import COLUMNS, render_comparison, rows_for_run
from .record import (
    RECORD_NAME,
    RECORD_SCHEMA_VERSION,
    VERDICT_NO_ARMS,
    VERDICT_SOURCE_CHANGED,
    VERDICT_VERIFIED,
    SourceGuard,
    close_guard,
    now,
    read_record,
    take_guard,
    write_record,
)

__all__ = [
    "COLUMNS",
    "RECORD_NAME",
    "RECORD_SCHEMA_VERSION",
    "VERDICT_NO_ARMS",
    "VERDICT_SOURCE_CHANGED",
    "VERDICT_VERIFIED",
    "SourceGuard",
    "close_guard",
    "now",
    "read_record",
    "render_comparison",
    "rows_for_run",
    "take_guard",
    "write_record",
]
