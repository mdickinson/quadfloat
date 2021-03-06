# Rounding directions.

_round_ties_to_even_offsets = [0, -1, -2, 1, 0, -1, 2, 1]


_FINITE, _INFINITE = False, True
_POSITIVE_ZERO, _NEGATIVE_ZERO = False, True


class RoundingDirection(object):
    def __init__(
            self, _rounder, _positive_overflow, _negative_overflow,
            _exact_zero_sum,
    ):
        self._rounder = _rounder
        self._positive_overflow = _positive_overflow
        self._negative_overflow = _negative_overflow
        self._exact_zero_sum = _exact_zero_sum

    def round_quarters(self, n, sign):
        """
        Round n / 4 to the nearest integer using this rounding direction.

        """
        return self._rounder(n, sign)

    def overflow_to_infinity(self, sign):
        """
        For a given sign, determine whether overflow maps to infinity or not.

        """
        if sign:
            return self._negative_overflow
        else:
            return self._positive_overflow

    def exact_zero_sum_sign(self):
        """
        Return the sign to use for an exact zero sum or difference.

        """
        return self._exact_zero_sum


round_ties_to_even = RoundingDirection(
    _rounder=lambda q, sign: q + _round_ties_to_even_offsets[q & 7] >> 2,
    _positive_overflow=_INFINITE,
    _negative_overflow=_INFINITE,
    _exact_zero_sum=_POSITIVE_ZERO,
)

round_ties_to_away = RoundingDirection(
    _rounder=lambda q, sign: (q + 2) >> 2,
    _positive_overflow=_INFINITE,
    _negative_overflow=_INFINITE,
    _exact_zero_sum=_POSITIVE_ZERO,
)

round_toward_positive = RoundingDirection(
    _rounder=lambda q, sign: q >> 2 if sign else -(-q >> 2),
    _positive_overflow=_INFINITE,
    _negative_overflow=_FINITE,
    _exact_zero_sum=_POSITIVE_ZERO,
)

round_toward_negative = RoundingDirection(
    _rounder=lambda q, sign: -(-q >> 2) if sign else q >> 2,
    _positive_overflow=_FINITE,
    _negative_overflow=_INFINITE,
    _exact_zero_sum=_NEGATIVE_ZERO,
)

round_toward_zero = RoundingDirection(
    _rounder=lambda q, sign: q >> 2,
    _positive_overflow=_FINITE,
    _negative_overflow=_FINITE,
    _exact_zero_sum=_POSITIVE_ZERO,
)
