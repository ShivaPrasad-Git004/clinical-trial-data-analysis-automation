
"""Shared parsing helpers for clinical trial data."""

import re

import numpy as np
import pandas as pd


# Conversion factors: one year = 365.25 days.
_UNIT_IN_YEARS = {
    "year": 1.0,
    "month": 1.0 / 12.0,
    "week": 7.0 / 365.25,
    "day": 1.0 / 365.25,
    "hour": 1.0 / (24.0 * 365.25),
    "minute": 1.0 / (24.0 * 60.0 * 365.25),
}

_ALIASES = {
    "yr": "year",
    "yrs": "year",
    "mo": "month",
    "mos": "month",
    "wk": "week",
    "wks": "week",
    "hr": "hour",
    "hrs": "hour",
    "min": "minute",
    "mins": "minute",
}

_AGE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)?\s*$"
)


def age_to_years(value):
    """Convert one age value to years; unknown values become NaN."""
    if value is None or value is pd.NA:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isfinite(value) and value >= 0:
            return float(value)
        return np.nan

    match = _AGE_RE.match(str(value))
    if not match:
        return np.nan

    number = float(match.group(1))
    unit = match.group(2)

    # A number without a unit is assumed to already be in years.
    if unit is None:
        return number

    unit = unit.lower()
    unit = _ALIASES.get(unit, unit)

    if unit.endswith("s") and unit[:-1] in _UNIT_IN_YEARS:
        unit = unit[:-1]

    factor = _UNIT_IN_YEARS.get(unit)

    if factor is None:
        return np.nan

    return number * factor


def ages_to_years(series):
    """Convert a complete age column to years."""
    return pd.Series(series).map(age_to_years).astype("float64")


def parse_dates(series):
    """Parse mixed-precision dates, including YYYY-MM dates."""
    s = pd.Series(series)

    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    s = s.astype("object").where(s.notna(), None)
    s = s.map(
        lambda v: None
        if v is None
        or (isinstance(v, str) and not v.strip())
        else v
    )

    return pd.to_datetime(
        s,
        errors="coerce",
        format="mixed",
    )


def read_csv_safe(path, **kwargs):
    """Preserve literal 'NA'; treat empty CSV fields as missing."""
    kwargs.setdefault("low_memory", False)

    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[""],
        **kwargs,
    )
