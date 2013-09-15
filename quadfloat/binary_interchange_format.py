# -*- coding: utf-8 -*-
from __future__ import absolute_import as _absolute_import
from __future__ import division as _division

import operator as _operator
import sys as _sys

from quadfloat.arithmetic import (
    _divide_to_odd,
    _isqrt,
    _remainder_nearest,
    _rshift_to_odd,
)
from quadfloat.attributes import (
    Attributes,
    get_current_attributes,
    partial_attributes,
    set_current_attributes,
    temporary_attributes,
)
from quadfloat.compat import (
    bit_length,
    builtins,
    _int_from_bytes,
    _int_to_bytes,
    _PyHASH_INF,
    _PyHASH_NINF,
    _PyHASH_NAN,
    _PyHASH_2INV,
    _PyHASH_MODULUS,
    STRING_TYPES,
    INTEGER_TYPES,
)
from quadfloat.exceptions import (
    InexactException,
    InvalidBooleanOperationException,
    InvalidIntegerOperationException,
    InvalidOperationException,
    OverflowException,
    UnderflowException,
    SignalingNaNException,
    DivideByZeroException,
)
from quadfloat.status_flags import (
    inexact as inexact_flag,
)
from quadfloat.interval import Interval as _Interval
from quadfloat.parsing import (
    parse_finite_decimal,
    parse_finite_hexadecimal,
    parse_infinity,
    parse_nan,
)
from quadfloat.rounding_direction import (
    round_ties_to_away,
    round_ties_to_even,
    round_toward_negative,
    round_toward_positive,
    round_toward_zero,
)
from quadfloat.tininess_detection import BEFORE_ROUNDING, AFTER_ROUNDING


# This module defines a function called 'abs', so we make sure to keep
# a reference to the builtin.
_abs = builtins.abs


def signal(exception):
    return exception.signal(get_current_attributes())


def exception_default_handler(exception, attributes):
    return exception.default_handler(attributes)


_default_flag_set = set()


_default_attributes = Attributes(
    rounding_direction=round_ties_to_even,
    tininess_detection=AFTER_ROUNDING,
    inexact_handler=exception_default_handler,
    invalid_operation_handler=exception_default_handler,
    overflow_handler=exception_default_handler,
    underflow_handler=exception_default_handler,
    divide_by_zero_handler=exception_default_handler,
    flag_set=_default_flag_set,
)

set_current_attributes(_default_attributes)


def compare_attributes():
    return Attributes(
        rounding_direction=round_toward_negative,
        tininess_detection=AFTER_ROUNDING,
        inexact_handler=exception_default_handler,
        overflow_handler=exception_default_handler,
        underflow_handler=exception_default_handler,
        flag_set=set(),
    )


# Attributes to use for conversions during comparison.
comparison_attributes = Attributes(
    rounding_direction=round_ties_to_even,
    tininess_detection=AFTER_ROUNDING,
    inexact_handler=exception_default_handler,
    flag_set=set(),
    # XXX Add test cases that need underflow, overflow;
    # divide_by_zero and invalid should be safe...
    # or should they?  What about signaling NaNs?
)


# Constants, utility functions.

_BINARY_INTERCHANGE_FORMAT_PRECISIONS = {
    16: 11,
    32: 24,
}


_FINITE = 'finite_type'
_INFINITE = 'infinite_type'
_NAN = 'nan_type'


# Conversions of Decimal numbers to strings.

def _decimal_format(sign, exponent, digits):
    # (-1)**sign * int(digits) * 10**exponent
    # Format in non-scientific form.

    assert not digits.startswith('0')
    assert not digits.endswith('0')

    if not digits:
        coefficient = '0'
    elif exponent >= 0:
        coefficient = digits + '0' * exponent
    elif exponent + len(digits) > 0:
        coefficient = digits[:exponent] + '.' + digits[exponent:]
    else:
        coefficient = '0.' + '0' * -(exponent + len(digits)) + digits
    return ('-' if sign else '') + coefficient


def _check_common_format(source1, source2):
    format = source1._format
    if source2._format != format:
        raise ValueError("Both operands should have the same format.")
    return format


# roundInexactToOdd is a useful primitive rounding direction for performing
# general rounding operations while avoiding problems from double rounding.
#
# The general pattern is: we want to compute a correctly rounded output for
# some mathematical function f, given zero or more inputs x1, x2, ...., and a
# rounding direction rnd, and a precision p.  Then:
#
#   (1) compute the correctly rounded output to precision p + 2 using
#       rounding-direction roundInexactToOdd.
#
#   (2) round the result of step 1 to the desired rounding direction `rnd` with
#   precision p.
#
# The roundInexactToOdd rounding direction has the property that for all the
# rounding directions we care about, the p + 2-bit result captures all the
# information necessary to rounding to any other rounding direction with p
# bits.  See the _divide_nearest function below for a nice example of this in
# practice.


class BinaryInterchangeFormat(object):
    """
    A BinaryInterchangeFormat instance represents one of the binary interchange
    formats described by IEEE 754-2008.  For example, the commonly-used
    double-precision binary floating-point type is given by
    BinaryInterchangeFormat(width=64):

    >>> binary64 = BinaryInterchangeFormat(width=64)

    Objects of this class should be treated as immutable.

    There are various attributes and read-only properties providing information
    about the format:

    >>> binary64.precision  # precision in bits
    53
    >>> binary64.width  # total width in bits
    64
    >>> binary64.emax  # maximum exponent
    1023
    >>> binary64.emin  # minimum exponent for normal numbers
    -1022

    Objects of this type are callable, and when called act like a class
    constructor to create floating-point numbers for the given format.

    >>> binary64('2.3')
    BinaryInterchangeFormat(width=64)('2.3')
    >>> str(binary64('2.3'))
    '2.3'

    """
    def __new__(cls, width):
        valid_width = width in (16, 32, 64) or width >= 128 and width % 32 == 0
        if not valid_width:
            raise ValueError(
                "Invalid width: {0}.  "
                "For an interchange format, width should be 16, 32, 64, "
                "or a multiple of 32 that's greater than 128.".format(width)
            )
        self = object.__new__(cls)
        self._width = int(width)
        return self

    def __repr__(self):
        return "BinaryInterchangeFormat(width={0})".format(self.width)

    def __str__(self):
        return "binary{0}".format(self.width)

    def __eq__(self, other):
        return self.width == other.width

    if _sys.version_info[0] == 2:
        # != is automatically inferred from == for Python 3.
        def __ne__(self, other):
            return not (self == other)

    def __hash__(self):
        return hash((BinaryInterchangeFormat, self.width))

    def __call__(self, *args, **kwargs):
        return self._from_value(*args, **kwargs)

    @property
    def width(self):
        return self._width

    @property
    def precision(self):
        if self.width in _BINARY_INTERCHANGE_FORMAT_PRECISIONS:
            return _BINARY_INTERCHANGE_FORMAT_PRECISIONS[self.width]
        else:
            # The IEEE 754 standard specifies that the precision should be
            # width - round(4 * log2(width)) + 13, where 'round' rounds to the
            # nearest integer.  The value of round(4 * log2(width)) can be
            # inferred from the number of bits needed to represent width ** 8.
            return self.width - bit_length(self.width ** 8) // 2 + 13

    @property
    def emax(self):
        return (1 << self.width - self.precision - 1) - 1

    @property
    def emin(self):
        return 1 - self.emax

    @property
    def qmin(self):
        return self.emin - self.precision + 1

    @property
    def qmax(self):
        return self.emax - self.precision + 1

    @property
    def _sign_bit(self):
        return 1 << self.width - 1

    @property
    def _exponent_bitmask(self):
        return (1 << self.width - 1) - (1 << self.precision - 1)

    @property
    def _significand_bitmask(self):
        return (1 << self.precision - 1) - 1

    @property
    def _quiet_bit(self):
        return 1 << self.precision - 2

    @property
    def _payload_bitmask(self):
        return (1 << self.precision - 2) - 1

    @property
    def _max_payload(self):
        """
        Maximum possible payload for a quiet or signaling NaN in this format.

        """
        return (1 << self.precision - 2) - 1

    @property
    def _min_normal_significand(self):
        """
        Minimum possible significand for a normal number.

        """
        return 1 << (self.precision - 1)

    @property
    def _max_significand(self):
        """
        Maximum possible significand.

        """
        return (1 << self.precision) - 1

    def _largest_finite(self, sign):
        """
        Largest representable finite value in this format, with the given sign.

        """
        exponent = self.qmax
        significand = self._max_significand
        return self._finite(sign, exponent, significand)

    def _from_value(self, value=0):
        """
        Float<nnn>([value])

        Create a new Float<nnn> instance from the given input.

        """
        if isinstance(value, _BinaryFloat):
            # Initialize from another _BinaryFloat instance.
            return self._from_binary_float(value)

        elif isinstance(value, float):
            # Initialize from a float.
            return self._from_float(value)

        elif isinstance(value, INTEGER_TYPES):
            # Initialize from an integer.
            return self._from_int(value)

        elif isinstance(value, STRING_TYPES):
            # Initialize from a string.
            return self.convert_from_decimal_character(value)

        else:
            raise TypeError(
                "Cannot construct a Float<nnn> instance from a "
                "value of type {0}".format(type(value))
            )

    def _from_binary_float(self, b):
        """
        Convert another _BinaryFloat instance to this format.

        """
        if b._type == _NAN:
            converted = self._from_nan_triple(
                sign=b._sign,
                signaling=b._signaling,
                payload=b._payload,
                clip_payload=True,
            )
        elif b._type == _INFINITE:
            # Infinities convert with no loss of information.
            converted = self._infinite(
                sign=b._sign,
            )
        else:
            # Finite value.
            converted = self._from_triple(
                sign=b._sign,
                exponent=b._exponent,
                significand=b._significand,
            )
        return converted

    def _from_int(self, n):
        """
        Convert the integer `n` to this format.

        """
        return self._from_triple(
            sign=n < 0,
            exponent=0,
            significand=_abs(n),
        )

    def _from_scaled_fraction(self, sign, exponent,
                              numerator, denominator):
        """
        Convert a number of the form +/-a/b * 2**e to this format.

        """
        a, b = numerator, denominator

        if a == 0:
            return self._zero(sign)

        # Find d such that 2 ** (d - 1) <= abs. value < 2 ** d.
        d = bit_length(a) - bit_length(b)
        d += (a >> d if d >= 0 else a << -d) >= b
        d += exponent

        exponent_out = max(d - self.precision - 2, self.qmin - 3)
        shift = exponent_out - exponent
        significand = _divide_to_odd(
            a << max(-shift, 0),
            b << max(0, shift),
        )
        return self._final_round(sign, exponent_out, significand)

    def _from_float(self, value):
        """
        Convert a float to this format.

        """
        # Convert via string.
        value = _binary64.convert_from_hex_character(value.hex())
        return self._from_binary_float(value)

    def _final_round(self, sign, e, q):
        """
        Make final rounding adjustment, using the current rounding-direction
        attribute.

        """
        attributes = get_current_attributes()

        # Debugging checks.  What's coming in should be in the auxiliary /
        # support format...

        # Auxiliary format: qmin = self.qmin - 3, precision = self.precision +
        # 2, no qmax, no infinities, no nans, conversion to this format always
        # does roundInexactToOdd.  Negative zero and subnormals are supported.

        assert (
            # normal in both auxiliary and output format; 2 extra bits
            e > self.qmin - 3 and bit_length(q) == self.precision + 2 or
            # normal in auxiliary, subnormal in output; 3 extra bits
            e == self.qmin - 3 and bit_length(q) == self.precision + 2 or
            # subnormal in both auxiliary and output format; 3 extra bits
            e == self.qmin - 3 and bit_length(q) < self.precision + 2
        )

        if q == 0:
            underflow = False
        elif bit_length(q) < self.precision + 2:
            underflow = True
        elif e == self.qmin - 3:
            if attributes.tininess_detection == BEFORE_ROUNDING:
                underflow = True
            elif attributes.tininess_detection == AFTER_ROUNDING:
                # Determine whether the result computed as though the exponent
                # range were unbounded would underflow.
                q2 = attributes.rounding_direction.round_quarters(q, sign)
                underflow = bit_length(q2) <= self.precision
            else:
                assert False, "never get here"  # pragma: no cover
        else:
            assert e > self.qmin - 3
            underflow = False

        # Remove extra bit in the subnormal case, using roundInexactToOdd.
        if e == self.qmin - 3:
            q, e = (q >> 1) | (q & 1), e + 1

        # Round
        rounding_direction = attributes.rounding_direction
        q2 = rounding_direction.round_quarters(q, sign)
        adj = (q2 << 2) - q
        q, e = q2, e + 2

        # Check whether we need to adjust the exponent.
        if bit_length(q) == self.precision + 1:
            q, e = q >> 1, e + 1

        # Signal the overflow exception when appropriate.
        if e > self.qmax:
            inexact = -1 if sign else 1
            if rounding_direction.overflow_to_infinity(sign):
                rounded = self._infinite(sign)
            else:
                rounded = self._largest_finite(sign)
            rounded = signal(OverflowException(rounded))
        else:
            inexact = (adj < 0) - (adj > 0) if sign else (adj > 0) - (adj < 0)
            rounded = self._finite(
                sign=sign,
                exponent=e,
                significand=q,
            )
            if underflow:
                rounded = signal(
                    UnderflowException(rounded, inexact != 0)
                )
            elif inexact:
                rounded = signal(InexactException(rounded))

        return rounded

    def _from_triple(self, sign, exponent, significand):
        """
        Round the value (-1) ** sign * significand * 2 ** exponent to the
        format 'self'.

        """
        if significand == 0:
            e = self.qmin - 3
        else:
            e = max(
                self.qmin - 3,
                exponent + bit_length(significand) - self.precision - 2,
            )
        q = _rshift_to_odd(significand, e - exponent)
        return self._final_round(sign, e, q)

    def _from_nan_triple(self, sign, signaling, payload, clip_payload=False):
        """
        Given a triple representing a NaN, convert to this format.
        Clip payload to within bounds if necessary.

        """
        min_payload = 1 if signaling else 0
        max_payload = self._max_payload
        if clip_payload:
            if payload < min_payload:
                payload = min_payload
            elif payload > max_payload:
                payload = max_payload
        else:
            if not min_payload <= payload <= max_payload:
                raise ValueError(
                    "{0} payload {1} is out of range for format {2}. "
                    "Valid range is {3} to {4}".format(
                        'sNaN' if signaling else 'NaN',
                        payload,
                        self,
                        min_payload,
                        max_payload,
                    )
                )

        return self._nan(
            sign=sign,
            signaling=signaling,
            payload=payload,
        )

    def _handle_nans(self, *sources):
        # Look for signaling NaNs.
        for source in sources:
            if source._type == _NAN and source._signaling:
                default_result = self._from_nan_triple(
                    sign=source._sign,
                    signaling=False,
                    payload=source._payload,
                    clip_payload=True,
                )
                exception = SignalingNaNException(default_result)
                return signal(exception)

        # All operands are quiet NaNs; return a result based on the first of
        # these.
        for source in sources:
            if source._type == _NAN:
                return self._from_nan_triple(
                    sign=source._sign,
                    signaling=source._signaling,
                    payload=source._payload,
                    clip_payload=True,
                )

        # If we get here, then _handle_nans has been called with all arguments
        # non-NaN.  This shouldn't happen.
        assert False, "never get here"  # pragma no cover

    def _handle_nans_min_max(self, source1, source2):
        # Handle NaNs in the manner required for min and max operations.

        # If we've got a combination of a quiet NaN and a non-NaN, return the
        # non-NaN.
        if source1._is_quiet() and source2._type != _NAN:
            return source2
        elif source2._is_quiet() and source1._type != _NAN:
            return source1
        else:
            return self._handle_nans(source1, source2)

    # Section 5.4.1: Arithmetic operations

    def addition(self, source1, source2):
        """
        Return 'source1 + source2', rounded to the format given by 'self'.

        """
        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        if source1._type == _INFINITE:
            if source2._type == _INFINITE and source1._sign != source2._sign:
                return self._handle_invalid()
            else:
                return self._infinite(source1._sign)

        if source2._type == _INFINITE:
            return self._infinite(source2._sign)

        exponent = min(source1._exponent, source2._exponent)
        significand = (
            (source1._signed_significand << source1._exponent - exponent) +
            (source2._signed_significand << source2._exponent - exponent)
        )
        if significand > 0:
            sign = False
        elif significand < 0:
            sign = True
        elif source1._sign == source2._sign:
            sign = source1._sign
        else:
            # For a zero result arising from different signs, the sign is
            # determined by the current rounding direction.
            attributes = get_current_attributes()
            sign = attributes.rounding_direction.exact_zero_sum_sign()

        return self._from_triple(
            sign=sign,
            exponent=exponent,
            significand=_abs(significand),
        )

    def subtraction(self, source1, source2):
        """
        Return 'source1 - source2', rounded to the format given by 'self'.

        """
        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        # For non-NaNs, subtraction(a, b) is equivalent to
        # addition(a, -b)
        return self.addition(source1, negate(source2))

    def multiplication(self, source1, source2):
        """
        Return 'source1 * source2', rounded to the format given by 'self'.

        """
        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        sign = source1._sign ^ source2._sign
        if source1._type == _INFINITE:
            if is_zero(source2):
                return self._handle_invalid()
            else:
                return self._infinite(sign=sign)

        if source2._type == _INFINITE:
            if is_zero(source1):
                return self._handle_invalid()
            else:
                return self._infinite(sign=sign)

        # finite * finite case.
        significand = source1._significand * source2._significand
        exponent = source1._exponent + source2._exponent
        return self._from_triple(
            sign=sign,
            exponent=exponent,
            significand=significand,
        )

    def division(self, source1, source2):
        """
        Return 'source1 / source2', rounded to the format given by 'self'.

        """
        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        sign = source1._sign ^ source2._sign

        # Handle infinities.
        if source1._type == _INFINITE:
            if source2._type == _INFINITE:
                return self._handle_invalid()
            else:
                return self._infinite(sign=sign)
        elif source2._type == _INFINITE:
            return self._zero(sign=sign)

        # Division by zero.
        if is_zero(source2):
            if is_zero(source1):
                return self._handle_invalid()
            else:
                return signal(DivideByZeroException(sign, self))

        # Finite / finite, with the denominator non-zero.
        return self._from_scaled_fraction(
            sign=sign,
            exponent=source1._exponent - source2._exponent,
            numerator=source1._significand,
            denominator=source2._significand,
        )

    def square_root(self, source1):
        """
        Return the square root of source1 in format 'self'.

        """
        if source1._type == _NAN:
            return self._handle_nans(source1)

        # sqrt(+-0) is +-0.
        if is_zero(source1):
            return self._zero(sign=source1._sign)

        # Any nonzero negative number is invalid.
        if source1._sign:
            return self._handle_invalid()

        # sqrt(+inf) -> +inf.
        if source1._type == _INFINITE and not source1._sign:
            return self._infinite(sign=False)

        sig = source1._significand
        exponent = source1._exponent

        # Exponent of result.
        d = (bit_length(sig) + exponent + 1) // 2
        e = max(d - self.precision - 2, self.qmin - 3)

        # Now find integer square root of sig, and add 1 if inexact.
        shift = exponent - 2 * e
        if shift >= 0:
            sig <<= shift
            rem = 0
        else:
            rem = sig & ~(-1 << -shift)
            sig >>= -shift
        q = _isqrt(sig)
        q |= bool(q * q != sig) or bool(rem)

        return self._final_round(False, e, q)

    def fused_multiply_add(self, source1, source2, source3):
        """
        Return source1 * source2 + source3, rounding once to format 'self'.

        """
        # Deal with any NaNs.
        any_nans = (
            source1._type == _NAN or
            source2._type == _NAN or
            source3._type == _NAN
        )
        if any_nans:
            return self._handle_nans(source1, source2, source3)

        sign12 = source1._sign ^ source2._sign

        # Deal with infinities in the first two arguments.
        if source1._type == _INFINITE:
            if is_zero(source2):
                return self._handle_invalid()
            else:
                return self.addition(self._infinite(sign12), source3)

        if source2._type == _INFINITE:
            if is_zero(source1):
                return self._handle_invalid()
            else:
                return self.addition(self._infinite(sign12), source3)

        # Deal with zeros in the first two arguments.
        if is_zero(source1) or is_zero(source2):
            return self.addition(self._zero(sign12), source3)

        # Infinite 3rd argument.
        if source3._type == _INFINITE:
            return self._infinite(source3._sign)

        # Multiply the first two arguments (both now finite and nonzero).
        significand12 = source1._significand * source2._significand
        exponent12 = source1._exponent + source2._exponent

        exponent = min(exponent12, source3._exponent)
        significand = (
            (significand12 * (-1) ** sign12 << exponent12 - exponent) +
            (source3._significand * (-1) ** source3._sign <<
             source3._exponent - exponent)
        )
        sign = (significand < 0 or
                significand == 0 and sign12 and source3._sign)

        return self._from_triple(
            sign=sign,
            exponent=exponent,
            significand=_abs(significand),
        )

    def convert_format(self, source):
        """
        Convert another _BinaryFloat instance to this format.

        """
        if source._type == _NAN:
            return self._handle_nans(source)
        return self._from_binary_float(source)

    def convert_from_int(self, n):
        """
        Convert the integer n to this format.

        """
        return self._from_int(n)

    def convert_to_hex_character_simple(self, source):
        """
        Convert the given source to this format.

        """
        # This function deliberately kept extremely simple, so that
        # it can serve as a basis for checking that two floats
        # are equivalent.
        if source._format != self:
            raise ValueError("Wrong format in convert_to_hex_character_simple")

        sign = '-' if source._sign else ''
        if source._type == _INFINITE:
            return '{sign}Infinity'.format(sign=sign)
        elif source._type == _NAN:
            return '{sign}{signaling}NaN({payload})'.format(
                sign=sign,
                signaling='s' if source._signaling else '',
                payload=source._payload,
            )
        elif source._type == _FINITE:
            return '{sign}0x{significand:x}p{exponent}'.format(
                sign=sign,
                significand=source._significand,
                exponent=source._exponent,
            )
        else:
            assert False, "never get here"  # pragma no cover

    def convert_from_decimal_character(self, s):
        """
        Convert the string s to this format.

        """
        # First attempt to interpret the string as a finite decimal.
        try:
            sign, exponent, significand = parse_finite_decimal(s)
        except ValueError:
            pass
        else:
            return self._from_scaled_fraction(
                sign=sign,
                exponent=exponent,
                numerator=significand * 5**max(exponent, 0),
                denominator=5**max(0, -exponent),
            )

        # Then as an infinity.
        try:
            sign = parse_infinity(s)
        except ValueError:
            pass
        else:
            return self._infinite(sign=sign)

        # Then as a NaN.
        try:
            sign, signaling, payload = parse_nan(s)
        except ValueError:
            pass
        else:
            if payload is None:
                payload = 1 if signaling else 0

            return self._from_nan_triple(
                sign=sign,
                signaling=signaling,
                payload=payload,
            )

        # And if all those failed, raise an exception.
        raise ValueError('invalid numeric decimal string: {0}'.format(s))

    def convert_from_hex_character(self, s):
        """
        Convert the string s to this format.

        """
        try:
            sign, exponent, significand = parse_finite_hexadecimal(s)
        except ValueError:
            pass
        else:
            return self._from_triple(
                sign=sign,
                exponent=exponent,
                significand=significand,
            )

        try:
            sign = parse_infinity(s)
        except ValueError:
            pass
        else:
            return self._infinite(sign=sign)

        try:
            sign, signaling, payload = parse_nan(s)
        except ValueError:
            pass
        else:
            if payload is None:
                payload = 1 if signaling else 0

            return self._from_nan_triple(
                sign=sign,
                signaling=signaling,
                payload=payload,
            )

        raise ValueError('invalid numeric string: {0}'.format(s))

    def _zero(self, sign):
        """
        Return a suitably-signed zero for this format.

        """
        return self._finite(
            sign=sign,
            exponent=self.qmin,
            significand=0,
        )

    def _infinite(self, sign):
        """
        Return a suitably-signed infinity for this format.

        """
        num = object.__new__(_BinaryFloat)
        num._format = self
        num._type = _INFINITE
        num._sign = bool(sign)
        return num

    def _nan(self, sign, signaling, payload):
        """
        Return a NaN for this format.

        """
        min_payload = 1 if signaling else 0
        if not min_payload <= payload <= self._max_payload:
            assert False, "NaN payload out of range"  # pragma no cover

        num = object.__new__(_BinaryFloat)
        num._format = self
        num._type = _NAN
        num._sign = bool(sign)
        num._signaling = bool(signaling)
        num._payload = int(payload)
        return num

    def _finite(self, sign, exponent, significand):
        """
        Return a finite number in this format.

        """
        # Check ranges of inputs.  Since this isn't (currently) part of the
        # public API, any error here is mine.  Hence the assert.
        if not self.qmin <= exponent <= self.qmax:
            assert False, "exponent out of range"  # pragma no cover
        if not 0 <= significand <= self._max_significand:
            assert False, "significand out of range"  # pragma no cover

        # Check normalization.
        normalized = (
            significand >= self._min_normal_significand or
            exponent == self.qmin
        )
        if not normalized:
            assert False, "non-normalized input to _finite"  # pragma no cover

        num = object.__new__(_BinaryFloat)
        num._format = self
        num._type = _FINITE
        num._sign = bool(sign)
        num._exponent = int(exponent)
        num._significand = int(significand)
        return num

    def _common_format(fmt1, fmt2):
        """
        Return the common BinaryInterchangeFormat suitable for mixed binary
        operations with operands of types 'fmt1' and 'fmt2'.

        fmt1 and fmt2 should be instances of BinaryInterchangeFormat.

        """
        return fmt1 if fmt1.width >= fmt2.width else fmt2

    def _handle_invalid(self):
        """
        Handle an invalid operation.

        """
        # XXX All uses of this function should be replaced with something
        # that signals a particular subclass of InvalidOperationException.
        return signal(InvalidOperationException(self))

    def _encode_as_int(self, source):
        """
        Encode 'source', which should have format 'self', as an unsigned int.

        """
        # Should only be used when 'source' has format 'self'.
        if not source._format == self:
            assert False, "shouldn't get here"  # pragma no cover

        if source._type == _NAN:
            result = source._payload + self._exponent_bitmask
            if not source._signaling:
                result += self._quiet_bit
        elif source._type == _INFINITE:
            result = self._exponent_bitmask
        else:
            # Note that this works for both normals and subnormals / zeros.
            exponent_field = source._exponent - self.qmin << self.precision - 1
            result = exponent_field + source._significand

        if source._sign:
            result += self._sign_bit

        return result

    def _decode_from_int(self, n):
        """
        Decode an unsigned int as encoded with _encode_as_int.

        """
        # Extract fields.
        significand = n & self._significand_bitmask
        exponent_field = n & self._exponent_bitmask
        sign = bool(n & self._sign_bit)

        # Construct value.
        if exponent_field == self._exponent_bitmask:
            # Infinity or NaN.
            if significand:
                # NaN.
                payload = significand & self._payload_bitmask
                signaling = not significand & self._quiet_bit
                return self._nan(sign, signaling, payload)
            else:
                # Infinity.
                return self._infinite(sign=sign)
        else:
            if exponent_field:
                # Normal number.
                exponent_field -= 1 << self.precision - 1
                significand += 1 << self.precision - 1
            exponent = (exponent_field >> self.precision - 1) + self.qmin
            return self._finite(sign, exponent, significand)

    def decode(self, encoded_value):
        """
        Decode a string of bytes to the corresponding Float<nnn> instance.

        """
        if len(encoded_value) != self.width // 8:
            raise ValueError("Wrong number of bytes for format.")

        n = _int_from_bytes(encoded_value)
        return self._decode_from_int(n)


_binary64 = BinaryInterchangeFormat(64)


class _BinaryFloat(object):
    @property
    def format(self):
        return self._format

    @property
    def _signed_significand(self):
        """
        Combination of self._sign and self._significand as an integer.

        """
        assert self._type == _FINITE
        return -self._significand if self._sign else self._significand

    def _shortest_decimal(self):
        """
        Convert to shortest Decimal instance that rounds back to the correct
        value.

        self should be finite.

        """
        assert self._type == _FINITE

        if self._significand == 0:
            return self._sign, 0, ''

        # General nonzero finite case.
        I = self._bounding_interval()
        exponent, digits = I.shortest_digit_string_floating()
        return self._sign, exponent, digits

    def _bounding_interval(self):
        """
        Interval of values rounding to self.

        Return the interval of real numbers that round to a nonzero finite self
        under 'round ties to even'.  This is used when computing decimal string
        representations of self.

        """
        is_boundary_case = (
            self._significand == self._format._min_normal_significand and
            self._exponent > self._format.qmin
        )

        if is_boundary_case:
            shift = self._exponent - 2
            high = (4 * self._significand + 2) << max(shift, 0)
            target = (4 * self._significand) << max(shift, 0)
            low = (4 * self._significand - 1) << max(shift, 0)
            denominator = 1 << max(0, -shift)
        else:
            shift = self._exponent - 1
            high = (2 * self._significand + 1) << max(shift, 0)
            target = (2 * self._significand) << max(shift, 0)
            low = (2 * self._significand - 1) << max(shift, 0)
            denominator = 1 << max(0, -shift)

        return _Interval(
            low=low,
            high=high,
            target=target,
            denominator=denominator,
            closed=self._significand % 2 == 0,
        )

    def _to_short_str(self):
        """
        Convert to a shortest Decimal string that rounds back to the given
        value.

        """
        # Quick returns for infinities and NaNs.
        if self._type == _INFINITE:
            return '-Infinity' if self._sign else 'Infinity'

        if self._type == _NAN:
            return '{sign}{signaling}NaN({payload})'.format(
                sign='-' if self._sign else '',
                signaling='s' if self._signaling else '',
                payload=self._payload,
            )

        # Finite case.
        sign, exponent, digits = self._shortest_decimal()
        return _decimal_format(sign, exponent, digits)

    def __repr__(self):
        return "{0!r}({1!r})".format(self._format, self._to_short_str())

    def __str__(self):
        return self._to_short_str()

    def _quieten_nan(self):
        """
        Quieten a NaN in this format.

        """
        assert is_signaling(self)
        return self._format._nan(
            sign=self._sign,
            signaling=False,
            payload=self._payload,
        )

    def _loud_copy(self):
        """
        Return a float identical to self, but re-signal any relevant
        exceptions.

        """
        # The only possible relevant exception is 'underflow'.  We don't
        # actually create a copy here: there's no need, since these
        # instances are immutable.
        if is_subnormal(self):
            exception = UnderflowException(self, False)
            return signal(exception)
        else:
            return self

    # IEEE 754 5.7.2: General operations.

    def _is_quiet(self):
        """
        Return True if self is a quiet NaN, and False otherwise.

        """
        return self._type == _NAN and not self._signaling

    def encode(self):
        """
        Encode a Float<nnn> instance as a bytestring.

        """
        equivalent_int = self._format._encode_as_int(self)
        return _int_to_bytes(equivalent_int, self._format.width // 8)

    def __pos__(self):
        return copy(self)

    def __neg__(self):
        return negate(self)

    def __abs__(self):
        return abs(self)

    def _convert_other(self, other):
        """
        Given numeric operands self and other, with self an instance of
        _BinaryFloat, convert other to an operand of type _BinaryFloat
        if necessary, and return the converted value.

        """
        # Convert other.
        if isinstance(other, _BinaryFloat):
            pass
        elif isinstance(other, float):
            other = _binary64._from_float(other)
        elif isinstance(other, INTEGER_TYPES):
            other = self._format._from_int(other)
        else:
            raise TypeError(
                "Can't convert operand {0} of type {1} to "
                "_BinaryFloat.".format(
                    other,
                    type(other),
                )
            )
        return other

    # Overloads for conversion to integer.
    def __int__(self):
        return convert_to_integer_toward_zero(self)

    if _sys.version_info[0] == 2:
        def __long__(self):
            return long(int(self))

    # Overload for conversion to float.
    def __float__(self):
        if self._type == _NAN:
            return float('-nan') if self._sign else float('nan')
        elif self._type == _INFINITE:
            return float('-inf') if self._sign else float('inf')
        else:
            a, b = self._significand, 1
            a, b = a << max(self._exponent, 0), b << max(0, -self._exponent)
            try:
                # True division, correctly rounded in Python >= 2.7.
                q = a / b
            except OverflowError:
                q = float('inf')
            return -q if self._sign else q

    # Binary arithmetic operator overloads.
    def __add__(self, other):
        other = self._convert_other(other)
        common_format = self._format._common_format(other._format)
        return common_format.addition(self, other)

    def __radd__(self, other):
        other = self._convert_other(other)
        common_format = self._format._common_format(other._format)
        return common_format.addition(other, self)

    def __sub__(self, other):
        other = self._convert_other(other)
        common_format = self._format._common_format(other._format)
        return common_format.subtraction(self, other)

    def __rsub__(self, other):
        other = self._convert_other(other)
        common_format = self._format._common_format(other._format)
        return common_format.subtraction(other, self)

    def __mul__(self, other):
        other = self._convert_other(other)
        common_format = self._format._common_format(other._format)
        return common_format.multiplication(self, other)

    def __rmul__(self, other):
        other = self._convert_other(other)
        common_format = self._format._common_format(other._format)
        return common_format.multiplication(other, self)

    def __truediv__(self, other):
        other = self._convert_other(other)
        common_format = self._format._common_format(other._format)
        return common_format.division(self, other)

    def __rtruediv__(self, other):
        other = self._convert_other(other)
        common_format = self._format._common_format(other._format)
        return common_format.division(other, self)

    if _sys.version_info[0] == 2:
        # Make sure that Python 2 divisions involving these types behave the
        # same way regardless of whether the division __future__ import is in
        # effect or not.
        __div__ = __truediv__
        __rdiv__ = __rtruediv__

    # Overloaded comparisons.

    def _rich_compare_general(self, other, operator, unordered_result):
        """
        Common code for __eq__, __ne__, __lt__, etc.

        `operator` is one of the 6 comparison operators from the operator
        module.

        `unordered_result` is the result that should be returned in the case
        that the comparison result is 'unordered', in the sense of 5.11.

        """
        if isinstance(other, INTEGER_TYPES):
            with temporary_attributes(compare_attributes()) as attributes:
                other = self._format._from_int(other)
                inexact = -1 if inexact_flag in attributes.flag_set else 0

        elif isinstance(other, float):
            with temporary_attributes(compare_attributes()) as attributes:
                other = self._format._from_float(other)
                inexact = -1 if inexact_flag in attributes.flag_set else 0

        elif isinstance(other, _BinaryFloat):
            with temporary_attributes(compare_attributes()) as attributes:
                other = self._format._from_binary_float(other)
                inexact = -1 if inexact_flag in attributes.flag_set else 0

        else:
            raise TypeError(
                "Can't convert operand {0} of type {1} to "
                "_BinaryFloat.".format(
                    other,
                    type(other),
                )
            )

        if is_signaling(self) or is_signaling(other):
            return _handle_invalid_bool(unordered_result)
        elif self._type == _NAN or other._type == _NAN:
            return unordered_result
        else:
            result = _compare_ordered(self, other) or inexact
            return operator(result, 0)

    if _sys.version_info[0] == 2:
        def __hash__(self):
            """
            Return hash value compatible with ints and floats.

            Raise TypeError for signaling NaNs.

            """
            if self._type == _NAN:
                if self._signaling:
                    raise ValueError('Signaling NaNs are unhashable.')
                return _PyHASH_NAN
            elif self._type == _INFINITE:
                return _PyHASH_NINF if self._sign else _PyHASH_INF
            else:
                # For Python 2, check whether the value matches that of
                # a Python int or float;  if so, use the hash of that.
                # We don't even try to get the hashes to match those
                # of Fraction or Decimal instances.

                # XXX. This is needlessly inefficient for huge values.
                if self == int(self):
                    return hash(int(self))
                elif self == float(self):
                    return hash(float(self))
    else:
        def __hash__(self):
            """
            Return hash value compatible with ints and floats.

            Raise TypeError for signaling NaNs.

            """
            if self._type == _NAN:
                if self._signaling:
                    raise ValueError('Signaling NaNs are unhashable.')
                return _PyHASH_NAN
            elif self._type == _INFINITE:
                return _PyHASH_NINF if self._sign else _PyHASH_INF
            else:
                # Assuming Python >= 3.2, compatibility with floats and ints
                # (not to mention Fraction and Decimal instances) follows if
                # we use the formulas described in the Python docs.
                base = 2 if self._exponent >= 0 else _PyHASH_2INV
                exp_hash = pow(base, _abs(self._exponent), _PyHASH_MODULUS)
                hash_ = self._significand * exp_hash % _PyHASH_MODULUS
                ans = -hash_ if self._sign else hash_

                return -2 if ans == -1 else ans

    def __eq__(self, other):
        return self._rich_compare_general(other, _operator.eq, False)

    def __ne__(self, other):
        return self._rich_compare_general(other, _operator.ne, True)

    def __lt__(self, other):
        return self._rich_compare_general(other, _operator.lt, False)

    def __gt__(self, other):
        return self._rich_compare_general(other, _operator.gt, False)

    def __le__(self, other):
        return self._rich_compare_general(other, _operator.le, False)

    def __ge__(self, other):
        return self._rich_compare_general(other, _operator.ge, False)


# 5.3: Homogeneous general-computational operations

# 5.3.1: General operations.

def _round_to_integral_general(self, rounding_direction, quiet):
    """
    General round_to_integral implementation used
    by the round_to_integral_* functions.

    """
    # NaNs.
    if self._type == _NAN:
        return self._format._handle_nans(self)

    # Infinities, zeros, and integral values are returned unchanged.
    if self._type == _INFINITE or is_zero(self) or self._exponent >= 0:
        return self

    # Round to a number of the form n / 4 using roundInexactToOdd.
    to_quarter = _rshift_to_odd(self._significand, -self._exponent - 2)

    # Then round to the nearest integer, using the prescribed rounding
    # direction.
    q = rounding_direction.round_quarters(to_quarter, self._sign)

    # Signal inexact if necessary.

    inexact = not quiet and q << 2 != to_quarter

    # Normalize.
    if q == 0:
        rounded = self._format._zero(self._sign)
    else:
        shift = self._format.precision - bit_length(q)
        rounded = self._format._finite(self._sign, -shift, q << shift)

    if inexact:
        return signal(InexactException(rounded))
    else:
        return rounded


def round_to_integral_ties_to_even(self):
    """
    Round self to an integral value in the same format, using
    the roundTiesToEven rounding direction.

    """
    return _round_to_integral_general(
        self,
        rounding_direction=round_ties_to_even,
        quiet=True,
    )


def round_to_integral_ties_to_away(self):
    """
    Round self to an integral value in the same format, using
    the roundTiesToAway rounding direction.

    """
    return _round_to_integral_general(
        self,
        rounding_direction=round_ties_to_away,
        quiet=True,
    )


def round_to_integral_toward_zero(self):
    """
    Round self to an integral value in the same format, using
    the roundTowardZero rounding direction.

    """
    return _round_to_integral_general(
        self,
        rounding_direction=round_toward_zero,
        quiet=True,
    )


def round_to_integral_toward_positive(self):
    """
    Round self to an integral value in the same format, using
    the roundTowardPositive rounding direction.

    In other words, this is the ceiling operation.

    """
    return _round_to_integral_general(
        self,
        rounding_direction=round_toward_positive,
        quiet=True,
    )


def round_to_integral_toward_negative(self):
    """
    Round self to an integral value in the same format, using
    the roundTowardNegative rounding direction.

    In other words, this is the floor operation.

    """
    return _round_to_integral_general(
        self,
        rounding_direction=round_toward_negative,
        quiet=True,
    )


def round_to_integral_exact(self):
    """
    Round self to an integral value using the current rounding-direction
    attribute.  Signal the 'inexact' exception if this changes the value.

    """
    rounding_direction = get_current_attributes().rounding_direction

    return _round_to_integral_general(
        self,
        rounding_direction=rounding_direction,
        quiet=False,
    )


def next_up(self):
    """
    Return the least floating-point number in the format of 'self'
    that compares greater than 'self'.

    """
    # NaNs follow the usual rules.
    if self._type == _NAN:
        return self._format._handle_nans(self)

    # Positive infinity maps to itself.
    if self._type == _INFINITE and not self._sign:
        return self

    # Negative zero is treated in the same way as positive zero.
    if is_zero(self) and self._sign:
        self = self._format._zero(sign=False)

    # Now we can cheat: encode as an integer, and then simply
    # increment or decrement the integer representation.
    n = self._format._encode_as_int(self)
    n += -1 if self._sign else 1
    return self._format._decode_from_int(n)


def next_down(self):
    """
    Return the greatest floating-point number in the format of 'self'
    that compares less than 'self'.

    """
    # NaNs follow the usual rules.
    if self._type == _NAN:
        return self._format._handle_nans(self)

    # Negative infinity maps to itself.
    if self._type == _INFINITE and self._sign:
        return self

    # Positive zero is treated in the same way as negative zero.
    if is_zero(self) and not self._sign:
        self = self._format._zero(sign=True)

    # Now we can cheat: encode as an integer, and then simply
    # increment or decrement the integer representation.
    n = self._format._encode_as_int(self)
    n += 1 if self._sign else -1
    return self._format._decode_from_int(n)


def remainder(self, other):
    """
    Defined as self - n * other, where n is the closest integer
    to the exact quotient self / other (with ties rounded to even).

    """
    # This is a homogeneous operation: both operands have the same format.
    format = _check_common_format(self, other)

    # NaNs follow the usual rules.
    if self._type == _NAN or other._type == _NAN:
        return format._handle_nans(self, other)

    # remainder(+/-inf, y) and remainder(x, 0) are invalid
    if self._type == _INFINITE or is_zero(other):
        return format._handle_invalid()

    # remainder(x, +/-inf) is x for any finite x.  Similarly, if x is
    # much smaller than y, remainder(x, y) is x.
    if other._type == _INFINITE or self._exponent <= other._exponent - 2:
        # Careful: we can't just return self here, since we have to
        # signal the underflow exception where appropriate.
        return self._loud_copy()
    else:
        # Now (other._exponent - exponent) is either 0 or 1, thanks to the
        # optimization above.
        exponent = min(self._exponent, other._exponent)
        modulus = other._significand << (other._exponent - exponent)
        # It's enough to compute modulo 2 * modulus, since the remainder
        # result is periodic modulo that value.
        multiplier = pow(2, self._exponent - exponent, 2 * modulus)
        remainder = _remainder_nearest(
            self._signed_significand * multiplier,
            modulus,
        )
        sign = self._sign if remainder == 0 else remainder < 0
        significand = _abs(remainder)

    # Normalize result.  It doesn't matter what rounding direction
    # we use, since the result should always be exact.
    with partial_attributes(
            rounding_direction=round_ties_to_even,
            tininess_detection=AFTER_ROUNDING,
    ):
        converted = format._from_triple(
            sign=sign,
            exponent=exponent,
            significand=significand,
        )
    return converted


def _min_max_num(self, other):
    """
    Helper function for min_num and max_num.  self and other
    should be non-NaN and have the same format.

    """
    if self._sign != other._sign:
        self_is_small = self._sign and not other._sign
    else:
        self_is_small = _compare_ordered(self, other) <= 0

    if self_is_small:
        return self, other
    else:
        return other, self


def _min_max_num_mag(self, other):
    """
    Helper function for min_num_mag and max_num_mag.  self and other
    should be non-NaN and have the same format.

    """
    cmp = _compare_ordered(abs(self), abs(other))
    if cmp != 0:
        self_is_small = cmp < 0
    elif self._sign != other._sign:
        self_is_small = self._sign and not other._sign
    else:
        # Identical; take the first.
        self_is_small = True

    if self_is_small:
        return self, other
    else:
        return other, self


def min_num(self, other):
    """
    Minimum of self and other.

    If self and other are differently-signed zeros, the negative zero is
    returned.

    """
    # This is a homogeneous operation: both operands have the same format.
    format = _check_common_format(self, other)

    # Special behaviour for NaNs:  if one operand is a quiet NaN and
    # the other is not, return the non-NaN operand.
    if self._type == _NAN or other._type == _NAN:
        return format._handle_nans_min_max(self, other)

    return _min_max_num(self, other)[0]._loud_copy()


def max_num(self, other):
    """
    Maximum of self and other.

    If self and other are differently-signed zeros, the positive zero is
    returned.

    """
    # This is a homogeneous operation: both operands have the same format.
    format = _check_common_format(self, other)

    # Special behaviour for NaNs:  if one operand is a quiet NaN and
    # the other is not, return the non-NaN operand.
    if self._type == _NAN or other._type == _NAN:
        return format._handle_nans_min_max(self, other)

    return _min_max_num(self, other)[1]._loud_copy()


def min_num_mag(self, other):
    """
    Minimum of self and other, by absolute value.

    If self and other are numerically equal (for example in the case of
    differently-signed zeros), self is returned.

    """
    # This is a homogeneous operation: both operands have the same format.
    format = _check_common_format(self, other)

    # Special behaviour for NaNs:  if one operand is a quiet NaN and
    # the other is not, return the non-NaN operand.
    if self._type == _NAN or other._type == _NAN:
        return format._handle_nans_min_max(self, other)

    return _min_max_num_mag(self, other)[0]._loud_copy()


def max_num_mag(self, other):
    """
    Maximum of self and other, by absolute value.

    If self and other are numerically equal (for example in the case of
    differently-signed zeros), other is returned.

    """
    # This is a homogeneous operation: both operands have the same format.
    format = _check_common_format(self, other)

    # Special behaviour for NaNs:  if one operand is a quiet NaN and
    # the other is not, return the non-NaN operand.
    if self._type == _NAN or other._type == _NAN:
        return format._handle_nans_min_max(self, other)

    return _min_max_num_mag(self, other)[1]._loud_copy()


# 5.3.3: logBFormat operations

def scale_b(self, n):
    """
    self * 2**n

    """
    # NaNs follow the usual rules.
    if self._type == _NAN:
        return self._format._handle_nans(self)

    # Infinities and zeros are unchanged.
    if self._type == _INFINITE or is_zero(self):
        return self

    # Finite case.
    return self._format._from_triple(
        sign=self._sign,
        exponent=self._exponent + n,
        significand=self._significand,
    )


def log_b(self):
    """
    exponent of self.

    """
    if self._type == _FINITE and self._significand != 0:
        # Finite nonzero case.
        return self._exponent + bit_length(self._significand) - 1

    # Exceptional cases.  The standard says: "logB(NaN), logB(∞), and
    # logB(0) return language-defined values outside the range ±2 × (emax +
    # p - 1) and signal the invalid operation exception."  The integer
    # format used for the output of logB is required to include the
    # range ±2 × (emax + p) inclusive.  If we stick within that range,
    # that leaves us 4 values to play with for the exceptional cases
    # of 0, infinity, and signaling and quiet NaNs.
    format = self._format
    limit = 2 * (format.emax + format.precision)

    # Zeros and infinities; return values that are at the appropriate
    # extremes of the finite result range.
    if is_zero(self):
        return _handle_invalid_int(-limit + 1)
    elif self._type == _INFINITE:
        return _handle_invalid_int(limit - 1)

    # NaNs.  We somewhat arbitrarily map signaling NaNs to the max
    # available value, and quiet NaNs to the min.
    assert self._type == _NAN
    if is_signaling(self):
        return _handle_invalid_int(limit)
    else:
        return _handle_invalid_int(-limit)


# 5.4.1 Arithmetic operations.

def _convert_to_integer_general(self, rounding_direction):
    if self._type == _NAN:
        # XXX Signaling nans should also raise the invalid operation
        # exception.
        raise ValueError("Cannot convert a NaN to an integer.")

    if self._type == _INFINITE:
        # NB. Python raises OverflowError here, which doesn't really
        # seem right.
        raise ValueError("Cannot convert an infinity to an integer.")

    # Round using roundInexactToOdd, with 2 extra bits.
    q = _rshift_to_odd(self._significand, -self._exponent - 2)
    q = rounding_direction.round_quarters(q, self._sign)

    # Use int() to convert from long if necessary
    return int(-q if self._sign else q)


def convert_to_integer_ties_to_even(self):
    """
    Round 'self' to the nearest Python integer, using the roundTiesToEven
    rounding direction.

    """
    return _convert_to_integer_general(
        self,
        rounding_direction=round_ties_to_even
    )


def convert_to_integer_toward_zero(self):
    """
    Round 'self' to a Python integer, using the roundTowardZero rounding
    direction.

    """
    return _convert_to_integer_general(
        self,
        rounding_direction=round_toward_zero
    )


def convert_to_integer_toward_positive(self):
    """
    Round 'self' to a Python integer, using the roundTowardPositive
    rounding direction.

    In other words, return the 'ceiling' of 'self' as a Python integer.

    """
    return _convert_to_integer_general(
        self,
        rounding_direction=round_toward_positive
    )


def convert_to_integer_toward_negative(self):
    """
    Round 'self' to a Python integer, using the roundTowardNegative
    rounding direction.

    In other words, return the 'floor' of 'self' as a Python integer.

    """
    return _convert_to_integer_general(
        self,
        rounding_direction=round_toward_negative
    )


def convert_to_integer_ties_to_away(self):
    """
    Round 'self' to the nearest Python integer, using the roundTiesToAway
    rounding direction.

    """
    return _convert_to_integer_general(
        self,
        rounding_direction=round_ties_to_away
    )


def _convert_to_integer_exact_general(self, rounding_direction):
    if self._type == _NAN:
        # XXX Signaling nans should also raise the invalid operation
        # exception.
        raise ValueError("Cannot convert a NaN to an integer.")

    if self._type == _INFINITE:
        # NB. Python raises OverflowError here, which doesn't really
        # seem right.
        raise ValueError("Cannot convert an infinity to an integer.")

    # Round using roundInexactToOdd, with 2 extra bits.
    quarters = _rshift_to_odd(self._significand, -self._exponent - 2)
    q = rounding_direction.round_quarters(quarters, self._sign)

    inexact = (q << 2) != quarters

    # Use int() to convert from long if necessary
    result = int(-q if self._sign else q)
    if inexact:
        return signal(InexactException(result))
    else:
        return result


def convert_to_integer_exact_ties_to_even(self):
    """
    Round 'self' to the nearest Python integer, using the roundTiesToEven
    rounding direction.

    """
    return _convert_to_integer_exact_general(
        self,
        rounding_direction=round_ties_to_even,
    )


def convert_to_integer_exact_ties_to_away(self):
    """
    Round 'self' to the nearest Python integer, using the roundTiesToAway
    rounding direction.

    """
    return _convert_to_integer_exact_general(
        self,
        rounding_direction=round_ties_to_away,
    )


def convert_to_integer_exact_toward_zero(self):
    """
    Round 'self' to the nearest Python integer, using the roundTowardZero
    rounding direction.

    """
    return _convert_to_integer_exact_general(
        self,
        rounding_direction=round_toward_zero,
    )


def convert_to_integer_exact_toward_positive(self):
    """
    Round 'self' to the nearest Python integer, using the roundTowardPositive
    rounding direction.

    """
    return _convert_to_integer_exact_general(
        self,
        rounding_direction=round_toward_positive,
    )


def convert_to_integer_exact_toward_negative(self):
    """
    Round 'self' to the nearest Python integer, using the roundTowardNegative
    rounding direction.

    """
    return _convert_to_integer_exact_general(
        self,
        rounding_direction=round_toward_negative,
    )


# 5.4.3: Conversion operations for binary formats.

def convert_to_hex_character(source, conversion_specification):
    """
    Convert the given binary float to a representative sequence,
    using information from the given conversion specification.

    """
    sign = '-' if source._sign else ''

    if source._type == _INFINITE:
        return '{sign}Infinity'.format(sign=sign)

    if source._type == _FINITE:
        trailing = source._format.precision - 1
        return '{sign}0x{first}.{rest:0{hex_digits}x}p{exp}'.format(
            sign=sign,
            first=source._significand >> trailing,
            rest=(source._significand & ~(-1 << trailing)) << (-trailing % 4),
            hex_digits=-(-trailing // 4),
            exp=source._exponent + trailing,
        )

    if source._type == _NAN:
        return '{sign}{signaling}NaN'.format(
            sign=sign,
            signaling='s' if source._signaling else '',
        )


# 5.6.1: Comparisons.

def _compare_ordered(source1, source2):
    """
    Given non-NaN values source1 and source2, compare them, returning -1, 0 or
    1 according as source1 < source2, source1 == source2, or source1 > source2.

    """
    # This function should only ever be called for two _BinaryFloat
    # instances with the same underlying format.
    assert source1._format == source2._format

    # Compare as though we've inverted the signs of both source1 and source2 if
    # necessary so that source1._sign is False.
    if is_zero(source1) and is_zero(source2):
        # Zeros are considered equal, regardless of sign.
        cmp = 0
    elif source1._sign != source2._sign:
        # Positive > negative.
        cmp = 1
    elif source1._type == _INFINITE:
        # inf > finite;  inf == inf
        cmp = 0 if source2._type == _INFINITE else 1
    elif source2._type == _INFINITE:
        # finite < inf
        cmp = -1
    elif source1._exponent != source2._exponent:
        cmp = -1 if source1._exponent < source2._exponent else 1
    elif source1._significand != source2._significand:
        cmp = -1 if source1._significand < source2._significand else 1
    else:
        cmp = 0

    return -cmp if source1._sign else cmp


def _handle_invalid_bool(default_bool):
    """
    This handler should be called when a function that would normally return a
    boolean signals invalid operation.  At the moment, the only such functions
    are comparisons involving a signaling NaN.

    """
    return signal(InvalidBooleanOperationException())


def _handle_invalid_int(payload):
    """
    This handler should be called when a function that would normally return an
    int signals invalid operation.

    """
    return signal(InvalidIntegerOperationException(payload))


def _compare_quiet_general(source1, source2, operator, unordered_result):
    """
    General quiet comparison implementation:  compare source1 and source2
    using the given operator in the case that neither source1 nor source2
    is a NaN, and returning the given unordered_result in the case that
    either source1 or source2 *is* a NaN.

    """
    if is_signaling(source1) or is_signaling(source2):
        return _handle_invalid_bool(unordered_result)
    elif source1._type == _NAN or source2._type == _NAN:
        return unordered_result
    else:
        with temporary_attributes(compare_attributes()) as attributes:
            source2 = source1._format._from_binary_float(source2)
            inexact = inexact_flag in attributes.flag_set

        result = _compare_ordered(source1, source2) or -inexact
        return operator(result, 0)


def _compare_signaling_general(source1, source2, operator, unordered_result):
    """
    General signaling comparison implementation:  compare source1 and source2
    using the given operator in the case that neither source1 nor source2
    is a NaN, and returning the given unordered_result in the case that
    either source1 or source2 *is* a NaN.

    """
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(unordered_result)
    else:
        fmt = source1._format
        with temporary_attributes(compare_attributes()) as attributes:
            source2 = fmt._from_binary_float(source2)
            inexact = inexact_flag in attributes.flag_set

        result = _compare_ordered(source1, source2) or -inexact
        return operator(result, 0)


def compare_quiet_equal(source1, source2):
    """
    Return True if source1 and source2 are numerically equal, else False.

    """
    return _compare_quiet_general(source1, source2, _operator.eq, False)


def compare_quiet_not_equal(source1, source2):
    """
    Return True if source1 and source2 are numerically equal, else False.

    """
    return _compare_quiet_general(source1, source2, _operator.ne, True)


def compare_quiet_greater(source1, source2):
    """
    Return True if source1 > source2, else False.

    """
    return _compare_quiet_general(source1, source2, _operator.gt, False)


def compare_quiet_greater_equal(source1, source2):
    """
    Return True if source1 >= source2, else False.

    """
    return _compare_quiet_general(source1, source2, _operator.ge, False)


def compare_quiet_less(source1, source2):
    """
    Return True if source1 < source2, else False.

    """
    return _compare_quiet_general(source1, source2, _operator.lt, False)


def compare_quiet_less_equal(source1, source2):
    """
    Return True if source1 <= source2, else False.

    """
    return _compare_quiet_general(source1, source2, _operator.le, False)


def compare_quiet_unordered(source1, source2):
    """
    Return True if either source1 or source2 is a NaN, else False.

    """
    operator = lambda x, y: False
    return _compare_quiet_general(source1, source2, operator, True)


def compare_quiet_not_greater(source1, source2):
    """
    Return True if source1 is not greater than source2, else False.

    Note that this function returns True if either source1 or source2 is a NaN.

    """
    return _compare_quiet_general(source1, source2, _operator.le, True)


def compare_quiet_less_unordered(source1, source2):
    """
    Return True if either source1 < source2, or source1 or source2 is a NaN.

    """
    return _compare_quiet_general(source1, source2, _operator.lt, True)


def compare_quiet_not_less(source1, source2):
    """
    Return True if source1 is not less than source2, else False.

    Note that this function returns True if either source1 or source2 is a NaN.

    """
    return _compare_quiet_general(source1, source2, _operator.ge, True)


def compare_quiet_greater_unordered(source1, source2):
    """
    Return True if either source1 > source2, or source1 or source2 is a NaN.

    """
    return _compare_quiet_general(source1, source2, _operator.gt, True)


def compare_quiet_ordered(source1, source2):
    """
    Return True if neither source1 nor source2 is a NaN.

    """
    operator = lambda x, y: True
    return _compare_quiet_general(source1, source2, operator, False)


def compare_signaling_equal(source1, source2):
    """
    Return True if source1 and source2 are numerically equal, else False.

    """
    return _compare_signaling_general(source1, source2, _operator.eq, False)


def compare_signaling_greater(source1, source2):
    """
    Return True if source1 > source2, else False.

    """
    return _compare_signaling_general(source1, source2, _operator.gt, False)


def compare_signaling_greater_equal(source1, source2):
    """
    Return True if source1 >= source2, else False.

    """
    return _compare_signaling_general(source1, source2, _operator.ge, False)


def compare_signaling_less(source1, source2):
    """
    Return True if source1 < source2, else False.

    """
    return _compare_signaling_general(source1, source2, _operator.lt, False)


def compare_signaling_less_equal(source1, source2):
    """
    Return True if source1 <= source2, else False.

    """
    return _compare_signaling_general(source1, source2, _operator.le, False)


def compare_signaling_not_equal(source1, source2):
    """
    Return True if source1 and source2 are numerically equal, else False.

    """
    return _compare_signaling_general(source1, source2, _operator.ne, True)


def compare_signaling_not_greater(source1, source2):
    """
    Return True if source1 is not greater than source2, else False.

    Note that this function returns True if either source1 or source2 is a NaN.

    """
    return _compare_signaling_general(source1, source2, _operator.le, True)


def compare_signaling_less_unordered(source1, source2):
    """
    Return True if either source1 < source2, or source1 or source2 is a NaN.

    """
    return _compare_signaling_general(source1, source2, _operator.lt, True)


def compare_signaling_not_less(source1, source2):
    """
    Return True if source1 is not less than source2, else False.

    Note that this function returns True if either source1 or source2 is a NaN.

    """
    return _compare_signaling_general(source1, source2, _operator.ge, True)


def compare_signaling_greater_unordered(source1, source2):
    """
    Return True if either source1 > source2, or source1 or source2 is a NaN.

    """
    return _compare_signaling_general(source1, source2, _operator.gt, True)


def _total_order_key(source):
    """
    Key function used to compare with total ordering.

    Assumes that the sign has already been dealt with.

    """
    if source._type == _FINITE:
        return 0, source._exponent, source._significand
    elif source._type == _INFINITE:
        return 1,
    else:
        assert source._type == _NAN
        return 2, not source._signaling, source._payload


def total_order(source1, source2):
    """
    Return True if source1 <= source2 under the total ordering specified by
    IEEE 754.

    """
    if not source1._format == source2._format:
        raise ValueError(
            "total_order operation not implemented for mixed formats."
        )

    if source1._sign != source2._sign:
        return source1._sign
    else:
        key1 = _total_order_key(source1)
        key2 = _total_order_key(source2)
        return key2 <= key1 if source1._sign else key1 <= key2


def total_order_mag(source1, source2):
    """
    Return True if abs(source1) <= abs(source2) under the total ordering
    specified by IEEE 754.

    """
    if not source1._format == source2._format:
        raise ValueError(
            "total_order operation not implemented for mixed formats."
        )
    return _total_order_key(source1) <= _total_order_key(source2)


signalingNaN = 'signalingNaN'
quietNaN = 'quietNaN'
negativeInfinity = 'negativeInfinity'
negativeNormal = 'negativeNormal'
negativeSubnormal = 'negativeSubnormal'
negativeZero = 'negativeZero'
positiveZero = 'positiveZero'
positiveSubnormal = 'positiveSubnormal'
positiveNormal = 'positiveNormal'
positiveInfinity = 'positiveInfinity'


def is_sign_minus(source):
    """
    Return True if source has a negative sign, else False.

    This applies to zeros and NaNs as well as infinities and nonzero finite
    numbers.

    """
    return source._sign


def is_normal(source):
    """
    Return True if source is normal and False otherwise.

    That is, return True if the source is not zero, subnormal, infinite or NaN.

    """
    return (
        source._type == _FINITE and
        source._format._min_normal_significand <= source._significand
    )


def is_finite(source):
    """
    Return True if source is finite; that is, zero, subnormal or normal (not
    infinite or NaN).

    """
    return source._type == _FINITE


def is_zero(source):
    """
    Return True if source is plus or minus 0.

    """
    return source._type == _FINITE and source._significand == 0


def is_subnormal(source):
    """
    Return True if source is subnormal, False otherwise.

    """
    return (
        source._type == _FINITE and
        0 < source._significand < source._format._min_normal_significand
    )


def is_infinite(source):
    """
    Return True if source is infinite, and False otherwise.

    """
    return source._type == _INFINITE


def is_nan(source):
    """
    Return True if source is a NaN, and False otherwise.

    """
    return source._type == _NAN


def is_signaling(source):
    """
    Return True if self is a signaling NaN, and False otherwise.

    """
    return source._type == _NAN and source._signaling


def is_canonical(source):
    """
    Return True if source is canonical.

    """
    # Currently no non-canonical values are supported.
    return True


def class_(source):
    """
    Determine which class a given number falls into.

    """
    if source._type == _NAN:
        if source._signaling:
            return signalingNaN
        else:
            return quietNaN
    elif source._type == _INFINITE:
        if source._sign:
            return negativeInfinity
        else:
            return positiveInfinity
    else:
        assert source._type == _FINITE
        fmt = source._format
        if source._significand == 0:
            if source._sign:
                return negativeZero
            else:
                return positiveZero
        elif source._significand < fmt._min_normal_significand:
            if source._sign:
                return negativeSubnormal
            else:
                return positiveSubnormal
        else:
            if source._sign:
                return negativeNormal
            else:
                return positiveNormal


def copy(source):
    """
    Return a copy of source.

    """
    if source._type == _FINITE:
        return source._format._finite(
            sign=source._sign,
            exponent=source._exponent,
            significand=source._significand,
        )
    elif source._type == _INFINITE:
        return source._format._infinite(
            sign=source._sign,
        )
    elif source._type == _NAN:
        return source._format._nan(
            sign=source._sign,
            signaling=source._signaling,
            payload=source._payload,
        )
    else:  # pragma: no cover
        assert False, "shouldn't get here"


def negate(source):
    """
    Return the negation of source.

    """
    if source._type == _FINITE:
        return source._format._finite(
            sign=not source._sign,
            exponent=source._exponent,
            significand=source._significand,
        )
    elif source._type == _INFINITE:
        return source._format._infinite(
            sign=not source._sign,
        )

    elif source._type == _NAN:
        return source._format._nan(
            sign=not source._sign,
            signaling=source._signaling,
            payload=source._payload,
        )
    else:  # pragma: no cover
        assert False, "shouldn't get here"


def abs(source):
    """
    Return the absolute value of source.

    """
    if source._type == _FINITE:
        return source._format._finite(
            sign=False,
            exponent=source._exponent,
            significand=source._significand,
        )
    elif source._type == _INFINITE:
        return source._format._infinite(
            sign=False,
        )
    elif source._type == _NAN:
        return source._format._nan(
            sign=False,
            signaling=source._signaling,
            payload=source._payload,
        )
    else:  # pragma: no cover
        assert False, "shouldn't get here"


def copy_sign(source1, source2):
    """
    Return a value with the same format as source1, but the sign bit of
    source2.

    source1 and source2 should have the same format.

    """
    # Currently implemented only as a homogeneous operation.
    if not source1._format == source2._format:
        raise ValueError(
            "copy_sign operation not implemented for mixed formats."
        )

    if source1._type == _FINITE:
        return source1._format._finite(
            sign=source2._sign,
            exponent=source1._exponent,
            significand=source1._significand,
        )
    elif source1._type == _INFINITE:
        return source1._format._infinite(
            sign=source2._sign,
        )
    elif source1._type == _NAN:
        return source1._format._nan(
            sign=source2._sign,
            signaling=source1._signaling,
            payload=source1._payload,
        )
    else:  # pragma: no cover
        assert False, "shouldn't get here"
