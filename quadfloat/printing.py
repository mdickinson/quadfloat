"""
Support for formatting of floating-point numbers as decimal and hexadecimal
strings.

"""


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

        # Whether to display an exponent or not.
        self.use_exponent = False

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
    def from_string(cls, conversion_specification):
        """
        Convert the given string to a conversion specification.

        """
        # XXX To do: define the conversion specification string.
        #
        # Current: conversion specification is a string of one
        # of the following forms:
        #   .6e -> round to 6 significant digits (round ties to even);
        #          the number may be any positive integer
        #   .6f -> round to 6 digits after the point (round ties to even)
        #          the number may be any integer (positive or negative)
        #   s -> shortest string that rounds back to the correct value.
        #   <empty string> -> shortest decimal giving the exact value
        #          of the binary number.
        cs = cls()
        if conversion_specification == 's':
            cs.style = 'shortest'
            cs.show_payload = True
        elif conversion_specification[:1] == '.':
            conversion_type = conversion_specification[-1]
            cs.min_fraction_length = int(conversion_specification[1:-1])
            if conversion_type == 'f':
                cs.min_exponent = -cs.min_fraction_length
            else:
                assert conversion_type == 'e'
                # e-style formatting: result is in scientific notation, and the
                # given number is the number of digits *after* the point.  But
                # there's always one digit before the point, too.
                cs.max_digits = cs.min_fraction_length + 1
                cs.use_exponent = True
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
        if self.use_exponent:
            return '{exponent_indicator}{sign}{exponent}'.format(
                exponent='{0:0{1}d}'.format(
                    abs(exponent), self.min_exponent_digits),
                exponent_indicator=self.exponent_indicator,
                sign='+' if exponent >= 0 else '-',
            )
        else:
            return ''

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
