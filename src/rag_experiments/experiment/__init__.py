"""What an experiment is, and how one arm of it is measured."""

from __future__ import annotations

from .checkout import Checkout, make_checkout
from .execute import ArmResult, run_arm
from .spec import (
    ARM_KINDS,
    CODE_ARM,
    SETTINGS_ARM,
    SPEC_SCHEMA_VERSION,
    Arm,
    RunSpec,
    load_spec,
)

__all__ = [
    "ARM_KINDS",
    "CODE_ARM",
    "SETTINGS_ARM",
    "SPEC_SCHEMA_VERSION",
    "Arm",
    "ArmResult",
    "Checkout",
    "RunSpec",
    "load_spec",
    "make_checkout",
    "run_arm",
]
