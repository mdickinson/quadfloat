# Rounding directions.

_round_ties_to_even_offsets = [0, -1, -2, 1, 0, -1, 2, 1]


class RoundingDirection(object):
    def __init__(self, _rounder):
        self._rounder = _rounder

    def round_quarters(self, n, sign):
        """
        Round n / 4 to the nearest integer using this rounding mode.

        """
        return self._rounder(n, sign)


round_ties_to_even = RoundingDirection(
    _rounder=lambda q, sign: q + _round_ties_to_even_offsets[q & 7] >> 2
)

round_ties_to_away = RoundingDirection(_rounder=lambda q, sign: (q + 2) >> 2)

round_toward_positive = RoundingDirection(
    _rounder=lambda q, sign: q >> 2 if sign else -(-q >> 2)
)

round_toward_negative = RoundingDirection(
    _rounder=lambda q, sign: -(-q >> 2) if sign else q >> 2
)

round_toward_zero = RoundingDirection(_rounder=lambda q, sign: q >> 2)
