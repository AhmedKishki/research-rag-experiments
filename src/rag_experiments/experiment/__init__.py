"""What an experiment is, and how one arm of it is measured."""

from __future__ import annotations

from .checkout import Checkout, make_checkout
from .execute import ArmFailure, ArmResult, PreparedArm, prepare_arm, run_arm
from .report_schema import ReportFacts, ReportValidationError, validate_report
from .spec import (
    ARM_KINDS,
    CODE_ARM,
    SETTINGS_ARM,
    SPEC_SCHEMA_VERSION,
    Arm,
    RunSpec,
    load_spec,
    segment,
)

__all__ = [
    "ARM_KINDS",
    "CODE_ARM",
    "SETTINGS_ARM",
    "SPEC_SCHEMA_VERSION",
    "Arm",
    "ArmFailure",
    "ArmResult",
    "Checkout",
    "PreparedArm",
    "ReportFacts",
    "ReportValidationError",
    "RunSpec",
    "load_spec",
    "make_checkout",
    "prepare_arm",
    "run_arm",
    "segment",
    "validate_report",
]
