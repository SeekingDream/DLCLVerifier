"""Exact binary32 oracle for tests: rational arithmetic, one rounding, IEEE signed zeros.

Independent of dlclv.primitives (it shares only `fpmodel.round_to_format`, the rational
rounding routine). Finite operands only."""
import math
from fractions import Fraction

import numpy as np

from dlclv.fpmodel import FORMATS, round_to_format

F32 = FORMATS["float32"]


def rn32(x: Fraction, zero_negative: bool = False) -> np.float32:
    """Round an exact rational to binary32. An exact zero takes the sign given by the caller."""
    if x == 0:
        return np.float32(-0.0 if zero_negative else 0.0)
    r = round_to_format(x, F32)
    if abs(r) > Fraction(F32.max):
        return np.float32(math.copysign(math.inf, x))
    if r == 0:                                   # underflow: the sign of the exact value
        return np.float32(math.copysign(0.0, x))
    return np.float32(float(r))


def neg(v) -> bool:
    return bool(np.signbit(np.float64(v)))


def fma(a, b, c) -> np.float32:
    """fl32(a * b + c), one rounding. Exact zero: -0 only if the product and c are both -0."""
    a, b, c = float(a), float(b), float(c)
    p = Fraction(a) * Fraction(b)
    p_neg = neg(a) != neg(b)
    return rn32(p + Fraction(c), zero_negative=(p == 0 and c == 0 and p_neg and neg(c)))


def mul(a, b) -> np.float32:
    a, b = float(a), float(b)
    return rn32(Fraction(a) * Fraction(b), zero_negative=neg(a) != neg(b))


def add(a, b) -> np.float32:
    a, b = float(a), float(b)
    return rn32(Fraction(a) + Fraction(b), zero_negative=(a == 0 and b == 0 and neg(a) and neg(b)))


def bits(x) -> bytes:
    return np.asarray(x, dtype=np.float32).tobytes()
