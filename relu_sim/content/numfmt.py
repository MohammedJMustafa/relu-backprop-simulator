"""Number formatting in the style of the homework solution.

* Values that are short exact decimals are written in full: 1.32, -0.3168, 0.61728.
* Anything else is rounded to 4 decimals (4 significant digits below 0.001)
  and the equation that shows it uses "≈".
* Columns of a table or vector share one number of decimals, as in Table 1.
* Minus signs are typographic (U+2212) and negative zero is printed as 0.
"""

from __future__ import annotations

import math
from typing import Iterable

MINUS = "−"
MAX_DECIMALS = 6
APPROX_DECIMALS = 4
_NOISE = 1e-14
_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _typographic(text: str) -> str:
    if text.startswith("-") and not any(ch in "123456789" for ch in text):
        text = text[1:]                      # "-0.0000" -> "0.0000"
    return text.replace("-", MINUS)


def decimals_needed(x: float, max_dp: int = MAX_DECIMALS) -> int | None:
    """Decimals needed to write ``x`` exactly, or None when more than ``max_dp`` are needed."""
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == 0.0:
        return 0
    if abs(x) < 10.0 ** -max_dp:
        return None
    tolerance = 1e-11 * max(1.0, abs(x))
    for dp in range(max_dp + 1):
        rounded = round(x, dp)
        if rounded != 0.0 and abs(x - rounded) <= tolerance:
            return dp
    return None


def is_exact(x: float, max_dp: int = MAX_DECIMALS) -> bool:
    """True when ``fmt`` writes the value without rounding (float noise around 0 counts as 0)."""
    return abs(float(x)) <= _NOISE or decimals_needed(x, max_dp) is not None


def fmt_sci(x: float, digits: int = 4) -> str:
    """Scientific notation such as 9.368×10⁻⁵."""
    mantissa, exponent = f"{x:.{digits - 1}e}".split("e")
    return _typographic(mantissa) + "×10" + str(int(exponent)).translate(_SUPERSCRIPTS)


def fmt(x: float, max_dp: int = MAX_DECIMALS) -> str:
    """Format like the homework (see the module docstring)."""
    x = float(x) + 0.0
    if math.isnan(x):
        return "undefined"
    if math.isinf(x):
        return "∞" if x > 0 else MINUS + "∞"
    if abs(x) <= _NOISE:
        return "0"
    dp = decimals_needed(x, max_dp)
    if dp is not None:
        return _typographic(f"{x:.{dp}f}")
    ax = abs(x)
    if ax >= 1e5 or ax < 1e-3:
        return fmt_sci(x)
    return _typographic(f"{x:.{APPROX_DECIMALS}f}")


def fmt_fixed(x: float, decimals: int) -> str:
    x = float(x) + 0.0
    if not math.isfinite(x):
        return fmt(x)
    return _typographic(f"{x:.{decimals}f}")


def is_exact_fixed(x: float, decimals: int) -> bool:
    return abs(x - round(x, decimals)) <= 1e-11 * max(1.0, abs(x))


def common_decimals(values: Iterable[float], max_dp: int = MAX_DECIMALS) -> int:
    """One number of decimals that writes every value of a column exactly (4 if impossible)."""
    needed = [decimals_needed(v, max_dp) for v in values]
    if any(n is None for n in needed):
        return APPROX_DECIMALS
    return max(needed, default=0)


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_percent(fraction: float, decimals: int = 1) -> str:
    return _typographic(f"{fraction * 100:.{decimals}f}") + " %"


def fmt_compact(x: float, digits: int = 4) -> str:
    """A short value for small labels (diagram chips): at most ``digits`` decimals."""
    x = float(x) + 0.0
    if not math.isfinite(x):
        return fmt(x)
    if x == 0.0:
        return "0"
    dp = decimals_needed(x, digits)
    if dp is not None:
        return _typographic(f"{x:.{dp}f}")
    if abs(x) < 10.0 ** -digits or abs(x) >= 1e5:
        return fmt_sci(x, 3)
    return _typographic(f"{x:.{digits}f}")
