"""
Representation of an interval with marked point.

The main application is to provide functionality to produce the shortest
decimal string corresponding to a number in an interval.

"""
from __future__ import absolute_import
from __future__ import division

from quadfloat.arithmetic import _divide_nearest
from quadfloat.compat import _map


class Interval(object):
    """
    Open or closed finite subinterval of the positive reals, with marked point.

    An instance of Interval represents an open or closed nonempty finite
    subinterval of the positive real numbers, with rational endpoints, along
    with a marked point on the real line.

    """
    def __new__(cls, low, high, target, denominator, closed):
        """
        Interval(low, high, target, denominator, closed) -> Interval object
        representing the open interval

            (low/denominator, high/denominator)

        if closed is False, and

            [low/denominator, high/denominator]

        if closed is True.  target/denominator gives the marked point inside
        the interval.

        """
        if not low < high:
            raise ValueError("low should be strictly less than high")
        if not denominator > 0:
            raise ValueError("denominator should be positive")

        self = object.__new__(cls)
        self.low = low
        self.high = high
        self.target = target
        self.denominator = denominator
        self.closed = closed
        return self

    def __mul__(self, n):
        """
        Scale this interval by a positive integer n.

        """
        if not n > 0:
            raise ValueError("n should be positive")

        return Interval(
            low=self.low * n,
            high=self.high * n,
            target=self.target * n,
            denominator=self.denominator,
            closed=self.closed,
        )

    def __truediv__(self, n):
        """
        Divide by a positive integer n.

        """
        if not n > 0:
            raise ValueError("n should be positive")

        return Interval(
            low=self.low,
            high=self.high,
            target=self.target,
            denominator=self.denominator * n,
            closed=self.closed,
        )


    def __sub__(self, n):
        """
        Subtract an integer from this interval.

        """
        nd = n * self.denominator
        return Interval(
            low=self.low - nd,
            high=self.high - nd,
            target=self.target - nd,
            denominator=self.denominator,
            closed=self.closed,
        )

    def __contains__(self, n):
        """
        Return True if this interval contains the given integer, else False.

        """
        I = self - n
        if I.closed:
            return I.low <= 0 <= I.high
        else:
            return I.low < 0 < I.high

    def high_integer(self):
        """
        Return greatest integer lying inside the upper bound of this interval.

        For a closed interval [low, high] this method returns the greatest
        integer n such that n <= high (that is, the floor of high).  For an
        open interval (low, high) it returns the greatest integer n such that
        n < high.

        Note that the returned integer is not necessarily inside the interval,
        since it may lie outside the lower bound.

        """
        if self.closed:
            return self.high // self.denominator
        else:
            return -(-self.high // self.denominator) - 1

    def low_integer(self):
        """
        Return least integer lying inside the lower bound of this interval.

        For a closed interval [low, high] this method returns the least integer
        n such that n >= low.  For an open interval (low, high) it returns the
        least integer n such that n > low.

        Note that the returned integer is not necessarily inside the interval,
        since it may lie outside the upper bound.

        """
        if self.closed:
            return -(-self.low // self.denominator)
        else:
            return self.low // self.denominator + 1

    def closest_integer_to_target(self):
        """
        Return closest integer to the target value inside the interval.

        """
        max_digit = self.high_integer()
        min_digit = self.low_integer()
        if min_digit > max_digit:
            raise ValueError("This interval contains no integers.")

        closest_digit = _divide_nearest(self.target, self.denominator)
        if closest_digit < min_digit:
            return min_digit
        elif closest_digit > max_digit:
            return max_digit
        else:
            return closest_digit

    def shortest_digit_string_fixed(self):
        """
        Given a subinterval of (0, 1), return the shortest string of digits
        such that 0.digits represents a point in this interval.

        In the case that there is more than one shortest string, return the one
        that's closest to the target value.

        """
        digits = []
        while True:
            self *= 10
            digit = self.high_integer()
            if digit in self:
                digits.append(self.closest_integer_to_target())
                break
            self -= digit
            digits.append(digit)

        return ''.join(_map(str, digits))

    def rescale_to_unit_interval(self):
        """
        Rescale this interval by a power of 10 so that it fits within (0, 1).

        self should lie entirely within the positive reals.

        Return an integer n and a new interval I such that self = 10**n * I.

        """
        n = len(str(self.high)) - len(str(self.denominator))
        I = self / 10 ** n if n >= 0 else self * 10 ** -n

        if I.high_integer() > 0:
            I /= 10
            n += 1

        # Check invariants.
        assert I.high_integer() == 0
        assert (I * 10).high_integer() > 0
        return n, I

    def shortest_digit_string_floating(self):
        """
        Generate the shortest string of digits representing a point
        in this interval.

        Return a pair (n, digits) such that int(digits) * 10**n gives the
        required point.

        """
        n, I = self.rescale_to_unit_interval()

        if 1 in I * 10:
            # Corner case: I contains 0.1, and possibly other powers of 10.
            # Rescale until the target value is in [1.0, 10.0).
            while I.target < I.denominator:
                n, I = n - 1, I * 10

            # Now target value is in [1.0, 10.0), so the two closest 1-digit
            # decimals to the target are both integers.
            digits = str(I.closest_integer_to_target())
            if digits.endswith('0'):
                n, digits = n + 1, digits[:-1]
        else:
            # Usual case: 0.1 not in I; use the fixed-point algorithm.
            digits = I.shortest_digit_string_fixed()
            n -= len(digits)
        return n, digits
