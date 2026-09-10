"""Display formatting utilities shared across marimo notebooks.

Provides reusable value → string formatters for timestamps, numeric
metrics, costs, and e-values. Each function follows a consistent
contract: accept a single value (possibly None), return a
display-ready string with em-dash ("\u2014") for missing data.
"""

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _detect_local_tz() -> ZoneInfo:
    """Detect the system's IANA timezone, falling back to UTC.

    Tries in order: $TZ env var, /etc/timezone, /etc/localtime symlink.
    """
    # $TZ environment variable — most explicit override.
    # POSIX allows a leading colon (e.g., ":America/New_York") — strip it.
    tz_env = os.environ.get("TZ")
    if tz_env:
        tz_clean = tz_env.removeprefix(":")
        if tz_clean:
            try:
                return ZoneInfo(tz_clean)
            except (KeyError, ValueError):
                pass

    # /etc/timezone — Debian/Ubuntu convention
    etc_tz = Path("/etc/timezone")
    if etc_tz.exists():
        try:
            return ZoneInfo(etc_tz.read_text().strip())
        except (KeyError, ValueError):
            pass

    # /etc/localtime symlink target — most Linux distros
    etc_lt = Path("/etc/localtime")
    if etc_lt.is_symlink():
        try:
            target = str(etc_lt.resolve())
            marker = "zoneinfo/"
            idx = target.index(marker)
            return ZoneInfo(target[idx + len(marker):])
        except (ValueError, KeyError):
            pass

    return ZoneInfo("UTC")


LOCAL_TZ: ZoneInfo = _detect_local_tz()


def format_error(e: BaseException) -> str:
    """Format an exception for display, falling back to repr when str is empty.

    Some exception types (e.g., httpx.TimeoutException, bare transport errors)
    return empty strings from str(). Using repr() as a fallback ensures the
    exception class name is always visible in error messages.
    """
    msg = str(e)
    return msg if msg else repr(e)


def format_timestamp(value: str | None, *, tz: ZoneInfo | None = None, empty: str = "—") -> str:
    """Convert an ISO 8601 UTC string to a localized, human-readable timestamp.

    When *tz* is None, uses the auto-detected LOCAL_TZ.
    Returns *empty* for None/empty input.

    Output format example: "Feb 15, 2026 05:15 PM EST"
    """
    if not value:
        return empty
    dt = datetime.fromisoformat(value)
    target_tz = tz if tz is not None else LOCAL_TZ
    local_dt = dt.astimezone(target_tz)
    return local_dt.strftime("%b %d, %Y %I:%M %p %Z")


# ── Numeric display formatters ───────────────────────────────────────
#
# Used as mo.ui.table format_mapping callbacks and for pre-formatting
# Plotly table cell values. Raw numeric types are preserved in table
# data for correct sorting; these functions control display only.


def format_e_value(v: float | None) -> str:
    """Fixed notation for readable values; scientific for extreme magnitudes.

    Values below 10,000 in absolute magnitude stay in fixed notation
    (at most 4 integer digits plus 3 decimals, so ``9999.999`` is the
    widest fixed-point rendering); anything larger switches to
    scientific notation so cell width stays bounded and the trailing
    decimals of a large value (for example ``500647.840``) never crowd
    out the informative leading digits. The threshold is chosen so the
    fixed-point maximum and the scientific minimum (``1.000e+04``) are
    visually matched — the format transitions smoothly rather than
    producing sudden width jumps at the boundary.
    """
    if v is None:
        return "\u2014"
    if abs(v) >= 10_000:
        return f"{v:.3e}"
    return f"{v:.3f}"


def format_accuracy(v: float | None) -> str:
    """Three-decimal accuracy or score, with em-dash for missing data."""
    if v is None:
        return "\u2014"
    return f"{v:.3f}"


def format_score_delta(v: float | None) -> str:
    """Signed three-decimal score difference, with em-dash for missing data.

    The metric family is a difference between unitless scores on [0, 1],
    such as balanced accuracy or a per-condition accuracy. The explicit
    sign is what distinguishes a difference from a level: the same
    ``.3f`` digits rendered unsigned read as a score rather than as a
    change, and the reader loses the direction. A score difference is a
    signed decimal, never "points" or percentage points, which belong to
    differences between percentage rates.
    """
    if v is None:
        return "\u2014"
    return f"{v:+.3f}"


def format_e_power(v: float | None) -> str:
    """Three-decimal empirical e-power (mean log e-value per accumulation step).

    The step is a batch for the per-cell folds and a betting round for the
    paired comparisons. Em-dash when no audit slices exist to average over.
    """
    if v is None:
        return "\u2014"
    return f"{v:.3f}"


def format_cost(v: float | None) -> str:
    """Dollar-formatted cost, or em-dash when pricing is unavailable."""
    if v is None:
        return "\u2014"
    return f"${v:.2f}"


def format_cost_per_trial(v: float | None) -> str:
    """Per-trial cost at four decimals ($0.0045), or em-dash when unpriced.

    The single source of the Cost per Trial ($) format:
    ``format_cost_cell``'s Cost per Trial ($) column and the cost
    scatter's axis labels both render through this, so a reader comparing
    figure to table sees one spelling of the same value.
    """
    if v is None:
        return "\u2014"
    return f"${v:.4f}"


def format_percent(v: float | None) -> str:
    """Fraction rendered as a human-readable percentage."""
    if v is None:
        return "\u2014"
    return f"{v:.1%}"


def format_optional_int(v: int | None) -> str:
    """Comma-formatted integer, or em-dash when the value is unavailable or zero.

    Uses a falsy check so that both None and 0 produce em-dash — appropriate
    for token counts where zero means the capability is absent (e.g., models
    without a reasoning trace report 0 reasoning tokens).
    """
    if not v:
        return "\u2014"
    return f"{v:,d}"
