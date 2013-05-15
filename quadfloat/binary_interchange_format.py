from __future__ import absolute_import as _absolute_import
from __future__ import division as _division

import math as _math
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
    _signal_invalid_operation,
    _signal_inexact,
    _signal_overflow,
    _signal_underflow,
    _current_rounding_direction,
)
from quadfloat.exceptions import (
    InexactException,
    InvalidBooleanOperationException,
    InvalidIntegerOperationException,
    InvalidOperationException,
    OverflowException,
    UnderflowException,
    SignalingNaNException,
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


_default_attributes = Attributes(
    rounding_direction=round_ties_to_even,
    tininess_detection=AFTER_ROUNDING,
)


def get_default_attributes():
    return _default_attributes


def get_current_attributes():
    # Current attributes are not currently modifiable...
    return _default_attributes


# Python 2 / 3 compatibility code.

if _sys.version_info.major == 2:
    _STRING_TYPES = basestring,
    _INTEGER_TYPES = (int, long)

    import binascii as _binascii

    def _int_to_bytes(n, length):
        return _binascii.unhexlify(format(n, '0{}x'.format(2 * length)))[::-1]

    def _int_from_bytes(bs):
        return int(_binascii.hexlify(bs[::-1]), 16)

    def _bytes_from_iterable(ns):
        """
        Create a bytestring from an iterable of integers.

        Each element of the iterable should be in range(256).

        """
        return ''.join(chr(n) for n in ns)

    # Values used to compute hashes.
    _PyHASH_INF = hash(float('inf'))
    _PyHASH_NINF = hash(float('-inf'))
    _PyHASH_NAN = hash(float('nan'))

else:
    _STRING_TYPES = str,
    _INTEGER_TYPES = int,
    _int_to_bytes = lambda n, length: n.to_bytes(length, byteorder='little')
    _int_from_bytes = lambda bs: int.from_bytes(bs, byteorder='little')
    _bytes_from_iterable = bytes

    _PyHASH_MODULUS = _sys.hash_info.modulus
    _PyHASH_2INV = pow(2, _PyHASH_MODULUS - 2, _PyHASH_MODULUS)
    _PyHASH_INF = _sys.hash_info.inf
    _PyHASH_NINF = -_sys.hash_info.inf
    _PyHASH_NAN = _sys.hash_info.nan


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


_round_ties_to_even_offsets = [0, -1, -2, 1, 0, -1, 2, 1]


# Round-to-odd is a useful primitive rounding direction for performing general
# rounding operations while avoiding problems from double rounding.
#
# The general pattern is: we want to compute a correctly rounded output for
# some mathematical function f, given zero or more inputs x1, x2, ...., and a
# rounding direction rnd, and a precision p.  Then:
#
#   (1) compute the correctly rounded output to precision p + 2 using
#       rounding-mode round-to-odd.
#
#   (2) round the result of step 1 to the desired rounding direction `rnd` with
#   precision p.
#
# The round-to-odd rounding direction has the property that for all the
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
        valid_width = width in {16, 32, 64} or width >= 128 and width % 32 == 0
        if not valid_width:
            raise ValueError(
                "Invalid width: {}.  "
                "For an interchange format, width should be 16, 32, 64, "
                "or a multiple of 32 that's greater than 128.".format(width)
            )
        self = object.__new__(cls)
        self._width = int(width)
        return self

    def __repr__(self):
        return "BinaryInterchangeFormat(width={})".format(self.width)

    def __str__(self):
        return "binary{}".format(self.width)

    def __eq__(self, other):
        return self.width == other.width

    if _sys.version_info.major == 2:
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
            return self.width - (self.width ** 8).bit_length() // 2 + 13

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

    def _from_value(self, value=0, attributes=None):
        """
        Float<nnn>([value])

        Create a new Float<nnn> instance from the given input.

        """
        if attributes is None:
            attributes = get_current_attributes()

        if isinstance(value, _BinaryFloat):
            # Initialize from another _BinaryFloat instance.
            return self._from_binary_float(value, attributes)[1]

        elif isinstance(value, float):
            # Initialize from a float.
            return self._from_float(value, attributes)[1]

        elif isinstance(value, _INTEGER_TYPES):
            # Initialize from an integer.
            return self._from_int(value, attributes)[1]

        elif isinstance(value, _STRING_TYPES):
            # Initialize from a string.
            return self._from_str(value, attributes)

        else:
            raise TypeError(
                "Cannot construct a Float<nnn> instance from a "
                "value of type {}".format(type(value))
            )

    def _from_binary_float(self, b, attributes):
        """
        Convert another _BinaryFloat instance to this format.

        """
        if b._type == _NAN:
            inexact = None
            converted = self._from_nan_triple(
                sign=b._sign,
                signaling=b._signaling,
                payload=b._payload,
            )
        elif b._type == _INFINITE:
            # Infinities convert with no loss of information.
            inexact = 0
            converted = self._infinite(
                sign=b._sign,
            )
        else:
            # Finite value.
            inexact, converted = self._from_triple(
                sign=b._sign,
                exponent=b._exponent,
                significand=b._significand,
                attributes=attributes,
            )
        return inexact, converted

    def _from_int(self, n, attributes):
        """
        Convert the integer `n` to this format.

        """
        return self._from_triple(
            sign=n < 0,
            exponent=0,
            significand=abs(n),
            attributes=attributes,
        )

    def _from_str(self, s, attributes):
        """
        Convert an input string to this format.

        """
        # Do we have a representation of a finite number?
        try:
            sign, exponent, significand = parse_finite_decimal(s)
        except ValueError:
            pass
        else:
            # Quick return for zeros.
            if significand == 0:
                return self._zero(sign)

            # Express (absolute value of) incoming string in form a / b;
            # find d such that 2 ** (d - 1) <= a / b < 2 ** d.
            a = significand * 5 ** max(exponent, 0)
            b = 5 ** max(0, -exponent)
            exp_diff = exponent
            d = a.bit_length() - b.bit_length()
            d += (a >> d if d >= 0 else a << -d) >= b
            d += exp_diff

            # Approximate a / b by number of the form q * 2 ** e.  We compute
            # two extra bits (hence the '- 2' below) of the result and round to
            # odd.
            exponent = max(d - self.precision - 2, self.qmin - 3)
            shift = exponent - exp_diff
            significand = _divide_to_odd(
                a << max(-shift, 0),
                b << max(0, shift),
            )
            return self._final_round(
                sign,
                exponent,
                significand,
                attributes,
            )[1]

        # Or a representation of infinity?
        try:
            sign = parse_infinity(s)
        except ValueError:
            pass
        else:
            return self._infinite(sign=sign)

        # Or a representation of a nan?
        try:
            sign, signaling, payload = parse_nan(s)
        except ValueError:
            pass
        else:
            return self._from_nan_triple(
                sign=sign,
                signaling=signaling,
                payload=payload,
            )

        raise ValueError('invalid numeric string: {}'.format(s))

    def _from_float(self, value, attributes):
        """
        Convert a float to this format.

        """
        sign = _math.copysign(1.0, value) < 0

        if _math.isnan(value):
            # XXX Consider trying to extract and transfer the payload here.
            inexact = None
            converted = self._nan(
                sign=sign,
                signaling=False,
                payload=0,
            )

        elif _math.isinf(value):
            # Infinities.
            inexact = 0
            converted = self._infinite(sign)
        elif value == 0.0:
            # Zeros
            inexact = 0
            converted = self._zero(sign)
        else:
            # Finite values.

            # Express absolute value of incoming float in format a / b;
            # find d such that 2 ** (d - 1) <= a / b < 2 ** d.
            a, b = abs(value).as_integer_ratio()
            d = a.bit_length() - b.bit_length()
            d += (a >> d if d >= 0 else a << -d) >= b

            # Approximate a / b by number of the form q * 2 ** e.  We compute
            # two extra bits (hence the '- 2' below) of the result and round to
            # odd.
            exponent = max(d - self.precision - 2, self.qmin - 3)
            significand = _divide_to_odd(
                a << max(-exponent, 0),
                b << max(exponent, 0),
            )
            inexact, converted = self._final_round(
                sign,
                exponent,
                significand,
                attributes,
            )

        return inexact, converted

    def _final_round(self, sign, e, q, attributes):
        """
        Make final rounding adjustment, using the rounding direction from the
        current context.  For now, only round-ties-to-even is supported.

        """
        # Debugging checks.  What's coming in should be in the auxiliary /
        # support format...

        # Auxiliary format: qmin = self.qmin - 3, precision = self.precision +
        # 2, no qmax, no infinities, no nans, conversion to this format always
        # does round-to-odd.  Negative zero and subnormals are supported.

        assert (
            # normal in both auxiliary and output format; 2 extra bits
            e > self.qmin - 3 and q.bit_length() == self.precision + 2 or
            # normal in auxiliary, subnormal in output; 3 extra bits
            e == self.qmin - 3 and q.bit_length() == self.precision + 2 or
            # subnormal in both auxiliary and output format; 3 extra bits
            e == self.qmin - 3 and q.bit_length() < self.precision + 2
        )

        if attributes.tininess_detection == BEFORE_ROUNDING:
            # Underflow *before* rounding.
            underflow = e == self.qmin - 3
        elif attributes.tininess_detection == AFTER_ROUNDING:
            # Underflow *after* rounding. The boundary is:
            #
            #    2**qmin * 2**(p - 1) * (1 - 2**(-p-1))
            #    = 2**qmin * (2**(p - 1) - 2**-2)
            #    = 2**(qmin - 3) * (2**(p + 2) - 2)
            #
            # Values *strictly* smaller than this trigger underflow.
            underflow = (
                e == self.qmin - 3 and
                0 < q < (1 << self.precision + 2) - 2
            )
        else:
            assert False, "never get here"  # pragma no cover

        # Remove extra bit in the subnormal case, using round-to-odd.
        if e == self.qmin - 3:
            q, e = (q >> 1) | (q & 1), e + 1

        # Round
        rounding_direction = attributes.rounding_direction
        q2 = rounding_direction.round_quarters(q, sign)
        adj = (q2 << 2) - q
        q, e = q2, e + 2

        # Check whether we need to adjust the exponent.
        if q.bit_length() == self.precision + 1:
            q, e = q >> 1, e + 1

        # Signal the overflow exception when appropriate.
        if e > self.qmax:
            inexact = -1 if sign else 1
            rounded = _signal_overflow(OverflowException(self._infinite(sign)))
        else:
            inexact = (adj < 0) - (adj > 0) if sign else (adj > 0) - (adj < 0)
            rounded = self._finite(
                sign=sign,
                exponent=e,
                significand=q,
            )
            if underflow:
                rounded = _signal_underflow(
                    UnderflowException(rounded, inexact != 0)
                )
            elif inexact:
                rounded = _signal_inexact(InexactException(rounded))

        return inexact, rounded

    def _from_triple(self, sign, exponent, significand, attributes):
        """
        Round the value (-1) ** sign * significand * 2 ** exponent to the
        format 'self'.

        """
        if significand == 0:
            return 0, self._zero(sign)

        d = exponent + significand.bit_length()

        # Find q such that q * 2 ** e approximates significand * 2 ** exponent.
        # Allow two extra bits for the final round.
        e = max(d - self.precision - 2, self.qmin - 3)
        q = _rshift_to_odd(significand, e - exponent)
        return self._final_round(sign, e, q, attributes)

    def _from_nan_triple(self, sign, signaling, payload):
        """
        Given a triple representing a NaN, convert to this format.
        Clip payload to within bounds if necessary.

        """
        payload = min(payload, self._max_payload)
        if signaling and payload == 0:
            payload = 1

        return self._nan(
            sign=sign,
            signaling=signaling,
            payload=payload,
        )

    def _handle_nans(self, *sources):
        # Look for signaling NaNs.
        for source in sources:
            if source._type == _NAN and source._signaling:
                exception = SignalingNaNException(self, source)
                return _signal_invalid_operation(exception)

        # All operands are quiet NaNs; return a result based on the first of
        # these.
        for source in sources:
            if source._type == _NAN:
                # Convert to this format if necessary.
                return self(source)

        # If we get here, then _handle_nans has been called with all arguments
        # non-NaN.  This shouldn't happen.
        assert False, "never get here"  # pragma no cover

    # Section 5.4.1: Arithmetic operations

    # 5.4.1 addition
    def addition(self, source1, source2, attributes=None):
        """
        Return 'source1 + source2', rounded to the format given by 'self'.

        """
        if attributes is None:
            attributes = get_current_attributes()

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
            (source1._significand * (-1) ** source1._sign <<
             source1._exponent - exponent) +
            (source2._significand * (-1) ** source2._sign <<
             source2._exponent - exponent)
        )
        sign = (significand < 0 or
                significand == 0 and source1._sign and source2._sign)

        return self._from_triple(
            sign=sign,
            exponent=exponent,
            significand=abs(significand),
            attributes=attributes,
        )[1]

    # 5.4.1 subtraction
    def subtraction(self, source1, source2, attributes=None):
        """
        Return 'source1 - source2', rounded to the format given by 'self'.

        """
        if attributes is None:
            attributes = get_current_attributes()

        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        # For non-NaNs, subtraction(a, b) is equivalent to
        # addition(a, b.negate())
        return self.addition(source1, source2.negate(), attributes)

    # 5.4.1 multiplication
    def multiplication(self, source1, source2, attributes=None):
        """
        Return 'source1 * source2', rounded to the format given by 'self'.

        """
        if attributes is None:
            attributes = get_current_attributes()

        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        sign = source1._sign ^ source2._sign
        if source1._type == _INFINITE:
            if source2.is_zero():
                return self._handle_invalid()
            else:
                return self._infinite(sign=sign)

        if source2._type == _INFINITE:
            if source1.is_zero():
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
            attributes=attributes,
        )[1]

    # 5.4.1 division
    def division(self, source1, source2, attributes=None):
        """
        Return 'source1 / source2', rounded to the format given by 'self'.

        """
        if attributes is None:
            attributes = get_current_attributes()

        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        sign = source1._sign ^ source2._sign
        if source1._type == _INFINITE:
            if source2._type == _INFINITE:
                return self._handle_invalid()
            else:
                return self._infinite(sign=sign)

        if source2._type == _INFINITE:
            # Already handled the case where source1 is infinite.
            return self._zero(sign=sign)

        if source1.is_zero():
            if source2.is_zero():
                return self._handle_invalid()
            else:
                return self._zero(sign=sign)

        if source2.is_zero():
            return self._infinite(sign=sign)

        # Finite / finite case.

        # First find d such that 2 ** (d-1) <= abs(source1) / abs(source2) <
        # 2 ** d.
        a = source1._significand
        b = source2._significand
        exp_diff = source1._exponent - source2._exponent
        d = a.bit_length() - b.bit_length()
        d += (a >> d if d >= 0 else a << -d) >= b
        d += exp_diff

        exponent = max(d - self.precision - 2, self.qmin - 3)
        shift = exponent - exp_diff
        significand = _divide_to_odd(
            a << max(-shift, 0),
            b << max(0, shift),
        )
        return self._final_round(sign, exponent, significand, attributes)[1]

    # 5.4.1 squareRoot
    def square_root(self, source1, attributes=None):
        """
        Return the square root of source1 in format 'self'.

        """
        if attributes is None:
            attributes = get_current_attributes()

        if source1._type == _NAN:
            return self._handle_nans(source1)

        # sqrt(+-0) is +-0.
        if source1.is_zero():
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
        d = (sig.bit_length() + exponent + 1) // 2
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

        return self._final_round(False, e, q, attributes)[1]

    # 5.4.1 fusedMultiplyAdd
    def fused_multiply_add(self, source1, source2, source3, attributes=None):
        """
        Return source1 * source2 + source3, rounding once to format 'self'.

        """
        if attributes is None:
            attributes = get_current_attributes()

        # Deal with any NaNs.
        if (source1._type == _NAN or source2._type == _NAN or
            source3._type == _NAN):
            return self._handle_nans(source1, source2, source3)

        sign12 = source1._sign ^ source2._sign

        # Deal with infinities in the first two arguments.
        if source1._type == _INFINITE:
            if source2.is_zero():
                return self._handle_invalid()
            else:
                return self.addition(self._infinite(sign12), source3)

        if source2._type == _INFINITE:
            if source1.is_zero():
                return self._handle_invalid()
            else:
                return self.addition(self._infinite(sign12), source3)

        # Deal with zeros in the first two arguments.
        if source1.is_zero() or source2.is_zero():
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
            significand=abs(significand),
            attributes=attributes,
        )[1]

    # 5.4.1 convertFromInt
    def convert_from_int(self, n, attributes=None):
        """
        Convert the integer n to this format.

        """
        if attributes is None:
            attributes = get_current_attributes()
        return self._from_int(n, attributes)[1]

    def convert_to_hex_character(self, source):
        """
        Convert the given source to this format.

        """
        # This function deliberately kept extremely simple, so that
        # it can serve as a basis for checking that two floats
        # are equivalent.
        if source._format != self:
            raise ValueError("Wrong format in convert_to_hex_character")

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

    def convert_from_hex_character(self, s, attributes=None):
        """
        Convert the string s to this format.

        """
        if attributes is None:
            attributes = get_current_attributes()

        try:
            sign, exponent, significand = parse_finite_hexadecimal(s)
        except ValueError:
            pass
        else:
            return self._from_triple(
                sign=sign,
                exponent=exponent,
                significand=significand,
                attributes=attributes,
            )[1]

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
            return self._from_nan_triple(
                sign=sign,
                signaling=signaling,
                payload=payload,
            )

        raise ValueError('invalid numeric string: {}'.format(s))

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
        if not 0 <= significand < 1 << self.precision:
            assert False, "significand out of range"  # pragma no cover

        # Check normalization.
        normalized = (
            significand >= 1 << self.precision - 1 or
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
        return _signal_invalid_operation(InvalidOperationException(self))

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
            self._significand == 1 << (self._format.precision - 1) and
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
        return "{!r}({!r})".format(self._format, self._to_short_str())

    def __str__(self):
        return self._to_short_str()

    def _quieten_nan(self):
        """
        Quieten a NaN in this format.

        """
        assert self.is_signaling()
        return self._format._nan(
            sign=self._sign,
            signaling=False,
            payload=self._payload,
        )

    # IEEE 754-2008 5.3.1: General operations.
    def _round_to_integral_general(self, rounding_direction, quiet):
        """
        General round_to_integral implementation used
        by the round_to_integral_* functions.

        """
        # NaNs.
        if self._type == _NAN:
            return self._format._handle_nans(self)

        # Infinities, zeros, and integral values are returned unchanged.
        if self._type == _INFINITE or self.is_zero() or self._exponent >= 0:
            return self

        # Round to a number of the form n / 4 using round-to-odd.
        to_quarter = _rshift_to_odd(self._significand, -self._exponent - 2)

        # Then round to the nearest integer, using the prescribed rounding
        # direction.
        q = rounding_direction._rounder(to_quarter, self._sign)

        # Signal inexact if necessary.

        inexact = not quiet and q << 2 != to_quarter

        # Normalize.
        if q == 0:
            rounded = self._format._zero(self._sign)
        else:
            shift = self._format.precision - q.bit_length()
            rounded = self._format._finite(self._sign, -shift, q << shift)

        if inexact:
            return _signal_inexact(InexactException(rounded))
        else:
            return rounded

    def round_to_integral_ties_to_even(self):
        """
        Round self to an integral value in the same format,
        with halfway cases rounding to even.

        """
        return self._round_to_integral_general(
            rounding_direction=round_ties_to_even,
            quiet=True,
        )

    def round_to_integral_ties_to_away(self):
        """
        Round self to an integral value in the same format,
        with halfway cases rounding away from zero.

        """
        return self._round_to_integral_general(
            rounding_direction=round_ties_to_away,
            quiet=True,
        )

    def round_to_integral_toward_zero(self):
        """
        Round self to an integral value in the same format,
        truncating any fractional part.

        """
        return self._round_to_integral_general(
            rounding_direction=round_toward_zero,
            quiet=True,
        )

    def round_to_integral_toward_positive(self):
        """
        Round self to an integral value in the same format,
        rounding non-exact integers to the next higher integer.

        In other words, this is the ceiling operation.

        """
        return self._round_to_integral_general(
            rounding_direction=round_toward_positive,
            quiet=True,
        )

    def round_to_integral_toward_negative(self):
        """
        Round self to an integral value in the same format,
        rounding non-exact integers to the next lower integer.

        In other words, this is the floor operation.

        """
        return self._round_to_integral_general(
            rounding_direction=round_toward_negative,
            quiet=True,
        )

    def round_to_integral_exact(self):
        """
        Round self to an integral value using the current rounding-direction
        attribute.  Signal the 'inexact' exception if this changes the value.

        """
        return self._round_to_integral_general(
            rounding_direction=_current_rounding_direction(),
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
        if self.is_zero() and self._sign:
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
        if self.is_zero() and not self._sign:
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
        if not self._format == other._format:
            raise ValueError(
                "remainder operation not implemented for mixed formats."
            )

        # NaNs follow the usual rules.
        if self._type == _NAN or other._type == _NAN:
            return self._format._handle_nans(self, other)

        # remainder(+/-inf, y) and remainder(x, 0) are invalid
        if self._type == _INFINITE or other.is_zero():
            return self._format._handle_invalid()

        # remainder(x, +/-inf) is x for any finite x.  Similarly, if x is
        # much smaller than y, remainder(x, y) is x.
        if other._type == _INFINITE or self._exponent <= other._exponent - 2:
            return self

        # Now (other._exponent - exponent) is either 0 or 1, thanks to the
        # optimization above.
        exponent = min(self._exponent, other._exponent)
        b = other._significand << (other._exponent - exponent)
        r = _remainder_nearest(
            self._significand * pow(2, self._exponent - exponent, 2 * b),
            b
        )
        sign = self._sign ^ (r < 0)
        significand = abs(r)

        # Normalize result.
        if significand == 0:
            return self._format._zero(sign)
        adjust = min(
            self._format.precision - significand.bit_length(),
            exponent - self._format.qmin,
        )
        return self._format._finite(
            sign,
            exponent - adjust,
            significand << adjust,
        )

    def min_num(self, other):
        """
        Minimum of self and other.

        If self and other are numerically equal (for example in the case of
        differently-signed zeros), self is returned.

        """
        # This is a homogeneous operation: both operands have the same format.
        if not self._format == other._format:
            raise ValueError(
                "min_num operation not implemented for mixed formats."
            )

        # Special behaviour for NaNs: if one operand is NaN and the other is
        # not then return the non-NaN operand.
        if self.is_nan() and not self.is_signaling() and not other.is_nan():
            return other
        if other.is_nan() and not other.is_signaling() and not self.is_nan():
            return self

        # Apart from the above special case, treat NaNs as normal.
        if self._type == _NAN or other._type == _NAN:
            return self._format._handle_nans(self, other)

        cmp = _compare_ordered(self, other)
        return self if cmp <= 0 else other

    def max_num(self, other):
        """
        Maximum of self and other.

        If self and other are numerically equal (for example in the case of
        differently-signed zeros), other is returned.

        """
        # This is a homogeneous operation: both operands have the same format.
        if not self._format == other._format:
            raise ValueError(
                "max_num operation not implemented for mixed formats."
            )

        # Special behaviour for NaNs: if one operand is NaN and the other is
        # not then return the non-NaN operand.
        if self.is_nan() and not self.is_signaling() and not other.is_nan():
            return other
        if other.is_nan() and not other.is_signaling() and not self.is_nan():
            return self

        # Apart from the above special case, treat NaNs as normal.
        if self._type == _NAN or other._type == _NAN:
            return self._format._handle_nans(self, other)

        cmp = _compare_ordered(self, other)
        return other if cmp <= 0 else self

    def min_num_mag(self, other):
        """
        Minimum of self and other, by absolute value.

        If self and other are numerically equal (for example in the case of
        differently-signed zeros), self is returned.

        """
        # This is a homogeneous operation: both operands have the same format.
        if not self._format == other._format:
            raise ValueError(
                "min_num_mag operation not implemented for mixed formats."
            )

        # Special behaviour for NaNs: if one operand is NaN and the other is
        # not then return the non-NaN operand.
        if self.is_nan() and not self.is_signaling() and not other.is_nan():
            return other
        if other.is_nan() and not other.is_signaling() and not self.is_nan():
            return self

        # Apart from the above special case, treat NaNs as normal.
        if self._type == _NAN or other._type == _NAN:
            return self._format._handle_nans(self, other)

        cmp = _compare_ordered(self.abs(), other.abs())
        return self if cmp <= 0 else other

    def max_num_mag(self, other):
        """
        Maximum of self and other, by absolute value.

        If self and other are numerically equal (for example in the case of
        differently-signed zeros), other is returned.

        """
        # This is a homogeneous operation: both operands have the same format.
        if not self._format == other._format:
            raise ValueError(
                "max_num_mag operation not implemented for mixed formats."
            )

        # Special behaviour for NaNs: if one operand is NaN and the other is
        # not then return the non-NaN operand.
        if self.is_nan() and not self.is_signaling() and not other.is_nan():
            return other
        if other.is_nan() and not other.is_signaling() and not self.is_nan():
            return self

        # Apart from the above special case, treat NaNs as normal.
        if self._type == _NAN or other._type == _NAN:
            return self._format._handle_nans(self, other)

        cmp = _compare_ordered(self.abs(), other.abs())
        return other if cmp <= 0 else self

    # IEEE 754 5.3.3: logBFormat operations
    def scale_b(self, n, attributes=None):
        """
        self * 2**n

        """
        if attributes is None:
            attributes = get_current_attributes()

        # NaNs follow the usual rules.
        if self._type == _NAN:
            return self._format._handle_nans(self)

        # Infinities and zeros are unchanged.
        if self._type == _INFINITE or self.is_zero():
            return self

        # Finite case.
        return self._format._from_triple(
            sign=self._sign,
            exponent=self._exponent + n,
            significand=self._significand,
            attributes=attributes,
        )[1]

    def log_b(self):
        """
        exponent of self.

        """
        if self._type == _NAN:
            if self.is_signaling():
                return _handle_invalid_int('signaling nan')
            else:
                return _handle_invalid_int('log_b(nan)')
        elif self._type == _INFINITE:
            return _handle_invalid_int('log_b(infinity)')
        elif self.is_zero():
            return _handle_invalid_int('log_b(zero)')

        # Finite nonzero case.
        return self._exponent + self._significand.bit_length() - 1

    # IEEE 754 5.7.2: General operations.

    def is_sign_minus(self):
        """
        Return True if self has a negative sign, else False.

        This applies to zeros and NaNs as well as infinities and nonzero finite
        numbers.

        """
        return self._sign

    def is_normal(self):
        """
        Return True if self is subnormal, False otherwise.

        """
        return (
            self._type == _FINITE and
            1 << self._format.precision - 1 <= self._significand
        )

    def is_finite(self):
        """
        Return True if self is finite; that is, zero, subnormal or normal (not
        infinite or NaN).

        """
        return self._type == _FINITE

    def is_zero(self):
        """
        Return True if self is plus or minus 0.

        """
        return (
            self._type == _FINITE and
            self._significand == 0
        )

    def is_subnormal(self):
        """
        Return True if self is subnormal, False otherwise.

        """
        return (
            self._type == _FINITE and
            0 < self._significand < 1 << self._format.precision - 1
        )

    def is_infinite(self):
        """
        Return True if self is infinite, and False otherwise.

        """
        return self._type == _INFINITE

    def is_nan(self):
        """
        Return True if self is a NaN, and False otherwise.

        """
        return self._type == _NAN

    def is_signaling(self):
        """
        Return True if self is a signaling NaN, and False otherwise.

        """
        return self._type == _NAN and self._signaling

    def is_canonical(self):
        """
        Return True if self is canonical.

        """
        # Currently no non-canonical values are supported.
        return True

    def encode(self):
        """
        Encode a Float<nnn> instance as a bytestring.

        """
        equivalent_int = self._format._encode_as_int(self)
        return _int_to_bytes(equivalent_int, self._format.width // 8)

    def copy(self):
        """
        Return a copy of self.

        """
        if self._type == _FINITE:
            return self._format._finite(
                sign=self._sign,
                exponent=self._exponent,
                significand=self._significand,
            )
        elif self._type == _INFINITE:
            return self._format._infinite(
                sign=self._sign,
            )
        elif self._type == _NAN:
            return self._format._nan(
                sign=self._sign,
                signaling=self._signaling,
                payload=self._payload,
            )
        else:  # pragma: no cover
            assert False, "shouldn't get here"

    def negate(self):
        """
        Return the negation of self.

        """
        if self._type == _FINITE:
            return self._format._finite(
                sign=not self._sign,
                exponent=self._exponent,
                significand=self._significand,
            )
        elif self._type == _INFINITE:
            return self._format._infinite(
                sign=not self._sign,
            )

        elif self._type == _NAN:
            return self._format._nan(
                sign=not self._sign,
                signaling=self._signaling,
                payload=self._payload,
            )
        else:  # pragma: no cover
            assert False, "shouldn't get here"

    def abs(self):
        """
        Return the absolute value of self.

        """
        if self._type == _FINITE:
            return self._format._finite(
                sign=False,
                exponent=self._exponent,
                significand=self._significand,
            )
        elif self._type == _INFINITE:
            return self._format._infinite(
                sign=False,
            )
        elif self._type == _NAN:
            return self._format._nan(
                sign=False,
                signaling=self._signaling,
                payload=self._payload,
            )
        else:  # pragma: no cover
            assert False, "shouldn't get here"

    def copy_sign(self, other):
        """
        Return a value with the same format as self, but the sign bit of other.

        """
        # Currently implemented only as a homogeneous operation.
        if not self._format == other._format:
            raise ValueError(
                "copy_sign operation not implemented for mixed formats."
            )

        if self._type == _FINITE:
            return self._format._finite(
                sign=other._sign,
                exponent=self._exponent,
                significand=self._significand,
            )
        elif self._type == _INFINITE:
            return self._format._infinite(
                sign=other._sign,
            )
        elif self._type == _NAN:
            return self._format._nan(
                sign=other._sign,
                signaling=self._signaling,
                payload=self._payload,
            )
        else:  # pragma: no cover
            assert False, "shouldn't get here"

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        return self.negate()

    def __abs__(self):
        return self.abs()

    def _convert_other(self, other, attributes):
        """
        Given numeric operands self and other, with self an instance of
        _BinaryFloat, convert other to an operand of type _BinaryFloat
        if necessary, and return the converted value.

        """
        # Convert other.
        if isinstance(other, _BinaryFloat):
            pass
        elif isinstance(other, float):
            other = _binary64._from_float(other, attributes)[1]
        elif isinstance(other, _INTEGER_TYPES):
            other = self._format._from_int(other, attributes)[1]
        else:
            raise TypeError(
                "Can't convert operand {} of type {} to "
                "_BinaryFloat.".format(
                    other,
                    type(other),
                )
            )
        return other

    # Overloads for conversion to integer.
    def __int__(self):
        return self.convert_to_integer_toward_zero()

    if _sys.version_info.major == 2:
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
        attributes = get_current_attributes()
        other = self._convert_other(other, attributes)
        # XXX attributes may play a role in determining what this common format
        # should be.  See e.g. 10.3 on preferredWidth attributes.
        common_format = self._format._common_format(other._format)
        return common_format.addition(self, other, attributes)

    def __radd__(self, other):
        attributes = get_current_attributes()
        other = self._convert_other(other, attributes)
        common_format = self._format._common_format(other._format)
        return common_format.addition(other, self)

    def __sub__(self, other):
        attributes = get_current_attributes()
        other = self._convert_other(other, attributes)
        common_format = self._format._common_format(other._format)
        return common_format.subtraction(self, other)

    def __rsub__(self, other):
        attributes = get_current_attributes()
        other = self._convert_other(other, attributes)
        common_format = self._format._common_format(other._format)
        return common_format.subtraction(other, self)

    def __mul__(self, other):
        attributes = get_current_attributes()
        other = self._convert_other(other, attributes)
        common_format = self._format._common_format(other._format)
        return common_format.multiplication(self, other)

    def __rmul__(self, other):
        attributes = get_current_attributes()
        other = self._convert_other(other, attributes)
        common_format = self._format._common_format(other._format)
        return common_format.multiplication(other, self)

    def __truediv__(self, other):
        attributes = get_current_attributes()
        other = self._convert_other(other, attributes)
        common_format = self._format._common_format(other._format)
        return common_format.division(self, other)

    def __rtruediv__(self, other):
        attributes = get_current_attributes()
        other = self._convert_other(other, attributes)
        common_format = self._format._common_format(other._format)
        return common_format.division(other, self)

    if _sys.version_info.major == 2:
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
        attributes = get_default_attributes()

        if isinstance(other, _INTEGER_TYPES):
            inexact, other = self._format._from_int(other, attributes)

        elif isinstance(other, float):
            inexact, other = self._format._from_float(other, attributes)

        elif isinstance(other, _BinaryFloat):
            inexact, other = self._format._from_binary_float(other, attributes)

        else:
            raise TypeError(
                "Can't convert operand {} of type {} to "
                "_BinaryFloat.".format(
                    other,
                    type(other),
                )
            )

        if self.is_signaling() or other.is_signaling():
            return _handle_invalid_bool(unordered_result)
        elif self._type == _NAN or other._type == _NAN:
            return unordered_result
        else:
            result = _compare_ordered(self, other) or inexact
            return operator(result, 0)

    if _sys.version_info.major == 2:
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
                exp_hash = pow(base, abs(self._exponent), _PyHASH_MODULUS)
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

    # 5.4.1 Arithmetic operations (conversions to integer).
    def _convert_to_integer_general(self, rounding_direction):
        if self._type == _NAN:
            # XXX Signaling nans should also raise the invalid operation
            # exception.
            raise ValueError("Cannot convert a NaN to an integer.")

        if self._type == _INFINITE:
            # NB. Python raises OverflowError here, which doesn't really
            # seem right.
            raise ValueError("Cannot convert an infinity to an integer.")

        # Round to odd, with 2 extra bits.
        q = _rshift_to_odd(self._significand, -self._exponent - 2)
        q = rounding_direction._rounder(q, self._sign)

        # Use int() to convert from long if necessary
        return int(-q if self._sign else q)

    def convert_to_integer_ties_to_even(self):
        """
        Round 'self' to the nearest Python integer, with halfway cases rounded
        to even.

        """
        return self._convert_to_integer_general(
            rounding_direction=round_ties_to_even
        )

    def convert_to_integer_toward_zero(self):
        """
        Round 'self' to a Python integer, truncating the fractional part.

        """
        return self._convert_to_integer_general(
            rounding_direction=round_toward_zero
        )

    def convert_to_integer_toward_positive(self):
        """
        Round 'self' to a Python integer, rounding toward positive infinity.

        In other words, return the 'ceiling' of 'self' as a Python integer.

        """
        return self._convert_to_integer_general(
            rounding_direction=round_toward_positive
        )

    def convert_to_integer_toward_negative(self):
        """
        Round 'self' to a Python integer, rounding toward negative infinity.

        In other words, return the 'floor' of 'self' as a Python integer.

        """
        return self._convert_to_integer_general(
            rounding_direction=round_toward_negative
        )

    def convert_to_integer_ties_to_away(self):
        """
        Round 'self' to the nearest Python integer, with halfway cases rounded
        away from zero.

        """
        return self._convert_to_integer_general(
            rounding_direction=round_ties_to_away
        )


# Section 5.6.1: Comparisons.

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
    if source1.is_zero() and source2.is_zero():
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
    return _signal_invalid_operation(InvalidBooleanOperationException())


def _handle_invalid_int(message):
    """
    This handler should be called when a function that would normally return an
    int signals invalid operation.

    """
    return _signal_invalid_operation(InvalidIntegerOperationException())


def _compare_quiet_general(source1, source2, operator, unordered_result):
    """
    General quiet comparison implementation:  compare source1 and source2
    using the given operator in the case that neither source1 nor source2
    is a NaN, and returning the given unordered_result in the case that
    either source1 or source2 *is* a NaN.

    """
    if source1.is_signaling() or source2.is_signaling():
        return _handle_invalid_bool(unordered_result)
    elif source1._type == _NAN or source2._type == _NAN:
        return unordered_result
    else:
        attributes = get_default_attributes()
        fmt = source1._format
        inexact, source2 = fmt._from_binary_float(source2, attributes)
        result = _compare_ordered(source1, source2) or inexact
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
        attributes = get_default_attributes()
        fmt = source1._format
        inexact, source2 = fmt._from_binary_float(source2, attributes)
        result = _compare_ordered(source1, source2) or inexact
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
