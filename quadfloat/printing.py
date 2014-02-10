"""Support for formatting of floating-point numbers as decimal and hexadecimal
strings.

Notes on conversion to shortest string:

There are a few competing desirable conditions for formatting of numbers in
the region of 2**precision.

0. All other considerations being equal, use non-scientific notation in
   preference to scientific notation, for the sake of familiarity and
   readability.
1. Don't give misleading information by padding with zeros in places where
   the true value doesn't have zeros.
2. Represent the consecutive integer range in non-exponential notation.
3. Have clear well-defined points for switching to exponential notation,
   preferably at a power-of-10 boundary.
4. Base the digits on the shortest string that rounds correctly.  This ensures
   decimal -> binary -> decimal roundtripping for values whose decimal
   significand has at most `q` digits, with q = floor(log10(2**(precision-1))).
5. For binary64, match the formatting used by Python.

Of the above: condition 0 is a general guideline.  Condition 1 is essential;
conditions 2 through 5 are 'nice to have' conditions.

For binary64 format, switching to exponential at 1e16 allows all of
conditions 1-5 to be satisfied.

However, for binary32 format these conditions are in conflict: all integers up
to 2**24 are representable, and 2**24 is 16777216, which lies between 10**7 and
10**8.  Between 2**24 and 2**25, only multiples of 2 are representable, and
between 2**25 and 2**26, only multiples of 4 are representable.  Now consider
the exactly representable number 2**26 + 8 == 67108872.  The shortest decimal
value that rounds correctly is 6710887e+1.  (Also consider 2**25 + 16, for a
similar example.  Note that between 2**24 and 2**25 there's no danger of
getting incorrect values when padding with zeros, since all multiples of 10 in
that range are exactly representable.)

So if we base our output on the shortest string (condition 4) and we avoid
scientific notation for numbers less than 1e8 (condition 0), we risk printing
67108870 for that value, which is misleading (condition 1).

Possible solutions:

  (A) switch to exponential notation at a power of 2 (2**24 or 2**25) instead
      of a power of 10.  This breaks condition 3.

  (B) for numbers in the range (10**7, 10**8), don't base the output on the
      shortest digit string, and instead give the correct last digit.  This
      breaks condition 4.

  (C) switch to exponential notation at 10**7.  This breaks condition 2.

Note: it's safe to use non-scientific notation for values less than 10**7, and
scientific notation for values not less than 10**8; it's the decade [10**7,
10**8) that requires special attention.

So: for values up to 2**24, we use the shortest representation (and pad with
zeros if necessary).  For values between 2**24 and 10**8, which are guaranteed
to be integers, we compute exact representations (round to exponent 0).

Generalizing to other formats: between 2**precision and the next highest power
of 10, we round exponent to 0.  This means that for binary64,

"""
from quadfloat.arithmetic import (
    _divide_to_odd,
    _rshift_to_odd,
)
from quadfloat.interval import Interval as _Interval
from quadfloat.rounding_direction import round_ties_to_even


_FINITE = 'finite_type'
_INFINITE = 'infinite_type'
_NAN = 'nan_type'


def _shortest_decimal(source):
    """
    Convert to shortest Decimal instance that rounds back to the correct
    value.

    source should be finite and nonzero.

    """
    assert source._type == _FINITE and source._significand > 0

    # General nonzero finite case.
    I = _bounding_interval(source)
    exponent, digits = I.shortest_digit_string_floating()
    return exponent, digits


def _bounding_interval(source):
    """
    Interval of values rounding to source.

    Return the interval of real numbers that round to a nonzero finite source
    under 'round ties to even'.  This is used when computing decimal string
    representations of source.

    """
    is_boundary_case = (
        source._significand == source._format._min_normal_significand and
        source._exponent > source._format.qmin
    )

    if is_boundary_case:
        shift = source._exponent - 2
        high = (4 * source._significand + 2) << max(shift, 0)
        target = (4 * source._significand) << max(shift, 0)
        low = (4 * source._significand - 1) << max(shift, 0)
        denominator = 1 << max(0, -shift)
    else:
        shift = source._exponent - 1
        high = (2 * source._significand + 1) << max(shift, 0)
        target = (2 * source._significand) << max(shift, 0)
        low = (2 * source._significand - 1) << max(shift, 0)
        denominator = 1 << max(0, -shift)

    return _Interval(
        low=low,
        high=high,
        target=target,
        denominator=denominator,
        closed=source._significand % 2 == 0,
    )


def compare_with_power_of_ten(source, n):
    """
    Return True iff abs(source) < 10**n.

    """
    shift = n - source._exponent
    lhs = source._significand * 5**max(0, -n) << max(0, -shift)
    rhs = 5**max(n, 0) << max(shift, 0)
    return lhs < rhs


def binary_hunt(predicate, start=0):
    """
    Find point at which a predicate changes from False to True.

    Given a predicate `predicate` defined for all integers such that:

       - predicate(n) is true for all sufficiently large n, and
       - predicate(n) is false for all sufficiently small n,

    find an integer n such that predicate(n-1) is false and predicate(n) is
    true.

    `start` should be a starting guess for the final value.  The closer it
    is, the quicker the search will return.

    """
    low, high = start - 1, start
    while predicate(low):
        low = 2 * low - high
    while not predicate(high):
        high = 2 * high - low
    while high - low > 1:
        mid = (high + low) // 2
        low, high = (low, mid) if predicate(mid) else (mid, high)
    return high


def _base_10_exponent(source):
    """
    Find the smallest integer n such that abs(source) < 10**n.

    Equivalently, find the unique integer n such that
    10**(n-1) <= abs(source) < 10**n, or n == 1 + floor(log10(abs(source))).

    Assumes that source is finite and nonzero.

    For numbers greater than or equal to 1.0, this amounts to
    counting the number of decimal digits in the integer part.

    """
    # Should only be called for nonzero finite numbers.
    assert source._type == _FINITE and source._significand != 0
    return binary_hunt(lambda n: compare_with_power_of_ten(source, n))


def _fix_decimal_exponent(source, places):
    """
    Convert the given float to the nearest number of the
    form +/- n * 10**places.

    """
    shift = source._exponent - places + 2
    if places > 0:
        q = _divide_to_odd(
            source._significand << max(shift, 0),
            5**places << max(0, -shift),
        )
    else:
        q = _rshift_to_odd(
            source._significand * 5**-places,
            -shift,
        )
    return places, round_ties_to_even.round_quarters(q, source._sign)


class ConversionSpecification(object):
    def __init__(self):
        # Overall style for printing.  This is either 'rounded' for
        # formatting based on rounding to a particular exponent, or
        # 'shortest' for formatting based on the shortest digit
        # string that rounds back to the correct number.
        self.style = 'rounded'

        # Minimum exponent for conversion of a nonzero finite number to
        # decimal.  Typically used for formatting with a fixed number of digits
        # after the decimal point.  None if there's no limit imposed.
        self.min_exponent = None

        # Maximum number of significand digits to use for conversion of a
        # nonzero finite number to decimal.  Typically used for 'e'-style
        # formatting.  None if there's no limit imposed.
        self.max_digits = None

        # Power of 10 below which to use scientific form, or None
        # to indicate that scientific form should never be used for
        # small values.
        self.lower_exponent_threshold = None

        # Power of 10 above which to use scientific form, or None
        # to indicate that scientific form should never be used for
        # large values.
        self.upper_exponent_threshold = None

        # Whether to use scientific notation for zeros
        self.scientific_zeros = False

        # Boolean indicating whether to show payloads on NaNs or not.
        # If true, NaNs will be formatted as e.g., 'NaN(123)'.  If false,
        # they're formatted simply as 'NaN'.
        self.show_payload = False

        # String to use for negative numbers.
        self.negative_sign = '-'

        # String to use for positive numbers.
        self.positive_sign = ''

        # String to use for decimal point.
        self.decimal_point = '.'

        # String to use for exponent prefix.
        self.exponent_indicator = 'e'

        # Minimum number of digits in printed exponent.
        self.min_exponent_digits = 2

        # String to use for infinities.
        self.infinity = 'Infinity'

        # String to use for quiet NaNs.
        self.qnan = 'NaN'

        # String to use for signaling NaNs.
        self.snan = 'sNaN'

        # Boolean indicating whether to force inclusion of a decimal point (so
        # that e.g. 123 is represented as '123.' rather than '123').
        self.force_decimal_point = False

        # Minimum number of digits following the decimal point.
        self.min_fraction_length = 0

        # Minimum number of digits preceding the decimal point.
        self.min_integral_length = 1

    @classmethod
    def from_string(cls, conversion_specification, format):
        """
        Convert the given string to a conversion specification.

        """
        # XXX To do: define the conversion specification string.
        #
        # Current: conversion specification is a string of one
        # of the following forms:
        #   .6e -> round to 7 significant digits (round ties to even);
        #          the number may be any positive integer
        #   .6f -> round to 6 digits after the point (round ties to even)
        #          the number may be any integer (positive or negative)
        #   .6g -> round to 6 significant digits; display result in
        #          scientific notation only for especially large or
        #          small numbers.
        #   s -> shortest string that rounds back to the correct value.
        #   <empty string> -> shortest decimal giving the exact value
        #          of the binary number.
        cs = cls()
        if conversion_specification == 's':
            cs.style = 'shortest'
            cs.show_payload = True
            cs.lower_exponent_threshold = -4
            cs.upper_exponent_threshold = format._pmin - 1
            cs.min_exponent_digits = 1
        elif conversion_specification[:1] == '.':
            conversion_type = conversion_specification[-1]
            if conversion_type == 'f':
                cs.min_fraction_length = int(conversion_specification[1:-1])
                cs.min_exponent = -cs.min_fraction_length
            elif conversion_type == 'g':
                cs.max_digits = int(conversion_specification[1:-1])
                cs.lower_exponent_threshold = -4
                cs.upper_exponent_threshold = cs.max_digits
            else:
                assert conversion_type == 'e'
                # e-style formatting: result is in scientific notation, and the
                # given number is the number of digits *after* the point.  But
                # there's always one digit before the point, too.
                cs.min_fraction_length = int(conversion_specification[1:-1])
                cs.max_digits = cs.min_fraction_length + 1
                cs.lower_exponent_threshold = 0
                cs.upper_exponent_threshold = 0
                cs.scientific_zeros = True
        else:
            assert conversion_specification == ''
        return cs

    def format_sign(self, sign):
        """
        Return a string giving the correct sign.

        On input, `sign` is a boolean: ``True`` for negative and ``False`` for
        positive.

        """
        return self.negative_sign if sign else self.positive_sign

    def format_finite_binary(self, source):
        """
        Format a finite binary number.

        """
        if source._significand == 0:
            exponent, digits = 0, 0
        elif self.style == 'shortest':
            # Some trickiness here: to avoid padding with misleading zeros, we
            # don't always use the shortest decimal.  For values greater than
            # or equal to 2**precision and less than the next higher power of
            # 10, we round to exponent 0 instead.
            use_simple = False
            if source._exponent > 0:
                if _base_10_exponent(source) < source._format._pmin:
                    use_simple = True
            if use_simple:
                exponent, digits = _fix_decimal_exponent(source, 0)
            else:
                exponent, digits = _shortest_decimal(source)
        else:
            target_exponent = min(source._exponent, 0)
            if self.min_exponent is not None:
                target_exponent = max(target_exponent, self.min_exponent)
            if self.max_digits is not None:
                e = _base_10_exponent(source) - self.max_digits
                target_exponent = max(target_exponent, e)
            exponent, digits = _fix_decimal_exponent(source, target_exponent)
        return self.format_finite(
            sign=source._sign,
            exponent=exponent,
            sdigits=str(digits) if digits else ''
        )

    def format_finite(self, sign, exponent, sdigits):
        """
        Format a finite number.

        """
        # Choose whether to use scientific notation or not.
        if sdigits:
            # 10**(adjusted_exponent - 1) <= abs(value) < 10**adjusted_exponent
            adjusted_exponent = exponent + len(sdigits)
            if (self.lower_exponent_threshold is not None and
                    adjusted_exponent <= self.lower_exponent_threshold):
                scientific = True
            elif (self.upper_exponent_threshold is not None and
                    adjusted_exponent > self.upper_exponent_threshold):
                scientific = True
            else:
                scientific = False
        else:
            scientific = self.scientific_zeros

        if scientific:
            display_exponent = exponent + len(sdigits) - 1 if sdigits else 0
            str_exponent = self.format_exponent(display_exponent)
        else:
            display_exponent = 0
            str_exponent = ''

        return '{sign}{significand}{exponent}'.format(
            sign=self.format_sign(sign),
            significand=self.place_decimal_point(
                sdigits, exponent - display_exponent),
            exponent=str_exponent,
        )

    def format_infinity(self, sign):
        """
        Format an infinity.

        """
        formatted_sign = self.format_sign(sign)
        return '{sign}{infinity}'.format(
            sign=formatted_sign,
            infinity=self.infinity,
        )

    def format_nan(self, sign, signaling, payload):
        """
        Format a NaN.

        """
        formatted_sign = self.format_sign(sign)
        if self.show_payload:
            return '{sign}{nan}({payload})'.format(
                sign=formatted_sign,
                nan=self.snan if signaling else self.qnan,
                payload=payload,
            )
        else:
            return '{sign}{nan}'.format(
                sign=formatted_sign,
                nan=self.snan if signaling else self.qnan,
            )

    def format_exponent(self, exponent):
        """
        Format the exponent.

        """
        return '{exponent_indicator}{sign}{exponent}'.format(
            exponent='{0:0{1}d}'.format(
                abs(exponent), self.min_exponent_digits),
            exponent_indicator=self.exponent_indicator,
            sign='+' if exponent >= 0 else '-',
        )

    def place_decimal_point(self, sdigits, exponent):
        """
        Place decimal point correctly within a digit string.

        Given a digit string `sdigits` and exponent `exponent`, representing
        the number ``int(sdigits) * 10**exponent``, return a string constructed
        from those digits and containing the decimal point in the correct
        position.  Pad the string with zeros on either the left or the right if
        necessary to reach the decimal point.

        >>> cs = ConversionSpecification()
        >>> cs.place_decimal_point('12345', 2)
        '1234500'
        >>> cs.place_decimal_point('12345', 0)
        '12345'
        >>> cs.place_decimal_point('12345', -2)
        '123.45'
        >>> cs.place_decimal_point('12345', -5)
        '0.12345'
        >>> cs.place_decimal_point('12345', -7)
        '0.0012345'
        >>> cs.place_decimal_point('0', 0)
        '0'
        >>> cs.place_decimal_point('', 0)
        '0'

        """
        # Find digits before and after the point.
        adjusted_exponent = exponent + len(sdigits)
        if exponent > 0:
            # Entire digit string is to the left of the point.
            digits_before = sdigits + '0' * exponent
            digits_after = ''
        elif adjusted_exponent >= 0:
            # Digit string includes or touches the point.
            digits_before = sdigits[:adjusted_exponent]
            digits_after = sdigits[adjusted_exponent:]
        else:
            # Entire digit string is to the right of the point.
            digits_before = ''
            digits_after = '0' * -adjusted_exponent + sdigits

        # Strip extraneous trailing zeros, but ensure minimum length
        # if necessary.
        digits_after = digits_after.rstrip('0')
        digits_after += '0' * (self.min_fraction_length - len(digits_after))

        # Strip extraneous leading zeros, and pad integral part to
        # minimum length.
        digits_before = digits_before.lstrip('0')
        digits_before = (
            '0' * (self.min_integral_length - len(digits_before)) +
            digits_before)

        # Corner case: ensure that we have at least one digit in the output.
        if not digits_before and not digits_after:
            digits_before = '0'

        if digits_after or self.force_decimal_point:
            return '{}{}{}'.format(
                digits_before,
                self.decimal_point,
                digits_after,
            )
        else:
            return digits_before
