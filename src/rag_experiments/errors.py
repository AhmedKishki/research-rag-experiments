"""The one failure this toolkit reports, and where to read the rest of it.

Every condition a caller must act on raises `ExperimentError` with a reason and,
where one exists, the command that fixes it. The command line turns it into the
exit status and the message, so no caller has to catch anything but this.
"""

from __future__ import annotations


class ExperimentError(RuntimeError):
    """A condition that stopped an experiment, and the reason it stopped.

    A measurement that cannot be taken is reported rather than approximated: a
    harness that substituted a default for a missing input would produce a number
    nobody could attribute.
    """


__all__ = ["ExperimentError"]
