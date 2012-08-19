"""
A collection of integer arithmetic primitives.

"""


_round_ties_to_even_offsets = [0, -1, -2, 1, 0, -1, 2, 1]


def _divide_to_odd(a, b):
    """
    Compute a / b.  Round inexact results to the nearest *odd* integer.

    >>> _divide_to_odd(-1, 4)
    -1
    >>> _divide_to_odd(0, 4)
    0
    >>> _divide_to_odd(1, 4)
    1
    >>> _divide_to_odd(4, 4)
    1
    >>> _divide_to_odd(5, 4)
    1
    >>> _divide_to_odd(7, 4)
    1
    >>> _divide_to_odd(8, 4)
    2

    """
    q, r = divmod(a, b)
    return q | bool(r)


def _rshift_to_odd(a, shift):
    """
    Compute a / 2**shift. Round inexact results to the nearest *odd* integer.

    >>> _rshift_to_odd(5, 2)
    1
    >>> _rshift_to_odd(6, 2)
    1
    >>> _rshift_to_odd(7, 2)
    1
    >>> _rshift_to_odd(8, 2)
    2
    >>> _rshift_to_odd(9, 2)
    3
    >>> _rshift_to_odd(5, -2)
    20

    """
    if shift <= 0:
        return a << -shift
    else:
        return (a >> shift) | bool(a & ~(-1 << shift))


def _divide_nearest(a, b):
    """
    Compute the nearest integer to the quotient a / b, rounding ties to the
    nearest even integer.  a and b should be integers, with b positive.

    >>> _divide_nearest(-14, 4)
    -4
    >>> _divide_nearest(-10, 4)
    -2
    >>> _divide_nearest(10, 4)
    2
    >>> _divide_nearest(14, 4)
    4
    >>> [_divide_nearest(i, 3) for i in range(-10, 10)]
    [-3, -3, -3, -2, -2, -2, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    >>> [_divide_nearest(i, 4) for i in range(-10, 10)]
    [-2, -2, -2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    >>> [_divide_nearest(i, 6) for i in range(-10, 10)]
    [-2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2]

    """
    q = _divide_to_odd(4 * a, b)
    return (q + _round_ties_to_even_offsets[q & 7]) >> 2


def _remainder_nearest(a, b):
    """
    Compute a - q * b where q is the nearest integer to the quotient a / b,
    rounding ties to the nearest even integer.  a and b should be integers,
    with b positive.

    Counterpart to divide_nearest.
    """
    return a - _divide_nearest(a, b) * b


def _isqrt(n):
    """
    Return the integer square root of n for any integer n >= 1.

    That is, return the unique integer m such that m*m <= n < (m+1)*(m+1).

    """
    m = n
    while True:
        q = n // m
        if m <= q:
            return m
        m = m + q >> 1
