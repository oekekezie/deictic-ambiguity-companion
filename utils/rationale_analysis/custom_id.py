"""Encode and parse rationale analysis custom_id values.

The rationale analysis custom_id encodes four dimensions —
example_id, config_key, batch_index, and trial — using ``__``
(double underscore) as the delimiter. This delimiter is safe because
example_id uses single ``_`` and config_key uses ``--``.

Format: ``ra__{example_id}__{config_key}__b{NN}__t{NNN}``
"""

import re


# Validates the batch segment (e.g., "b00", "b01")
_BATCH_RE: re.Pattern[str] = re.compile(r"^b(\d{2,})$")
# Validates the trial segment (e.g., "t001", "t042")
_TRIAL_RE: re.Pattern[str] = re.compile(r"^t(\d{3,})$")


def encode_rationale_analysis_custom_id(
    example_id: str,
    config_key: str,
    batch_index: int,
    trial: int,
) -> str:
    """Encode trial identity into a rationale analysis custom_id.

    Validates that neither example_id nor config_key contains the ``__``
    delimiter, which would break the split-based parser. Raises ValueError
    on invalid inputs.
    """
    if "__" in example_id:
        raise ValueError(
            f"example_id {example_id!r} contains '__' which would corrupt "
            f"the custom_id delimiter structure"
        )
    if "__" in config_key:
        raise ValueError(
            f"config_key {config_key!r} contains '__' which would corrupt "
            f"the custom_id delimiter structure"
        )
    if batch_index < 0:
        raise ValueError(f"batch_index must be non-negative, got {batch_index}")
    if trial < 1:
        raise ValueError(f"trial must be >= 1, got {trial}")

    # Minimum 3-digit trial width, matching expand_trials() convention
    trial_width = max(3, len(str(trial)))
    return (
        f"ra__{example_id}__{config_key}"
        f"__b{batch_index:02d}__t{trial:0{trial_width}d}"
    )


def parse_rationale_analysis_custom_id(
    custom_id: str,
) -> tuple[str, str, int, int]:
    """Parse a rationale analysis custom_id into its four components.

    Returns (example_id, config_key, batch_index, trial).

    Expects the assembly-time custom_id — before expand_trials() appends
    ``_trial_{KKK}``. Use ``parse_custom_id()`` from
    ``utils.batch_inference.processed`` to strip that suffix first.
    """
    parts = custom_id.split("__")
    if len(parts) != 5:
        raise ValueError(
            f"Expected 5 '__'-delimited segments in rationale analysis "
            f"custom_id, got {len(parts)}: {custom_id!r}"
        )

    prefix, example_id, config_key, batch_part, trial_part = parts

    if prefix != "ra":
        raise ValueError(
            f"Rationale analysis custom_id must start with 'ra__', "
            f"got prefix {prefix!r}: {custom_id!r}"
        )

    batch_match = _BATCH_RE.match(batch_part)
    if batch_match is None:
        raise ValueError(
            f"Invalid batch segment {batch_part!r} in custom_id {custom_id!r}, "
            f"expected 'b' followed by 2+ digits"
        )

    trial_match = _TRIAL_RE.match(trial_part)
    if trial_match is None:
        raise ValueError(
            f"Invalid trial segment {trial_part!r} in custom_id {custom_id!r}, "
            f"expected 't' followed by 3+ digits"
        )

    return example_id, config_key, int(batch_match.group(1)), int(trial_match.group(1))
