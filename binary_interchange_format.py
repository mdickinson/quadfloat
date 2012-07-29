from __future__ import division as _division

import contextlib as _contextlib
import decimal as _decimal
import math as _math
import operator as _operator
import re as _re
import sys as _sys


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

    # Iterator version of zip.
    from future_builtins import zip as _zip

    # Values used to compute hashes.
    if _sys.maxint == 2 ** 31 - 1:
        _PyHASH_MODULUS = 2 ** 31 - 1
    elif _sys.maxint == 2 ** 63 - 1:
        _PyHASH_MODULUS == 2 ** 61 - 1
    _PyHASH_2INV = pow(2, _PyHASH_MODULUS - 2, _PyHASH_MODULUS)
    _PyHASH_INF = hash(float('inf'))
    _PyHASH_NINF = hash(float('-inf'))
    _PyHASH_NAN = hash(float('nan'))

else:
    _STRING_TYPES = str,
    _INTEGER_TYPES = int,
    _int_to_bytes = lambda n, length: n.to_bytes(length, byteorder='little')
    _int_from_bytes = lambda bs: int.from_bytes(bs, byteorder='little')
    _bytes_from_iterable = bytes

    _zip = zip

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


_number_parser = _re.compile(r"""        # A numeric string consists of:
    (?P<sign>[-+])?                     # an optional sign, then either ...
    (?:
        (?P<finite>                     # a finite number
            (?=\d|\.\d)                 # with at least one digit
            (?P<int>\d*)                # having a (maybe empty) integer part
            (?:\.(?P<frac>\d*))?        # and a (maybe empty) fractional part
            (?:E(?P<exp>[-+]?\d+))?     # and an optional exponent, or ...
        )
    |
        (?P<infinite>                   # ... an infinity, or ...
            Inf(?:inity)?
        )
    |
        (?P<nan>
            (?P<signaling>s)?           # ... an (optionally signaling)
            NaN                         # NaN
            (?:\((?P<payload>\d+)\))?   # with optional parenthesized payload.
        )
    )
    \Z
""", _re.VERBOSE | _re.IGNORECASE).match


_round_ties_to_even_offsets = [0, -1, -2, 1, 0, -1, 2, 1]


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


# Round-to-odd is a useful primitive rounding mode for performing general
# rounding operations while avoiding problems from double rounding.
#
# The general pattern is: we want to compute a correctly rounded output for
# some mathematical function f, given zero or more inputs x1, x2, ...., and a
# rounding mode rnd, and a precision p.  Then:
#
#   (1) compute the correctly rounded output to precision p + 2 using
#       rounding-mode round-to-odd.
#
#   (2) round the result of step 1 to the desired rounding mode `rnd` with
#   precision p.
#
# The round-to-odd rounding mode has the property that for all the rounding
# modes we care about, the p + 2-bit result captures all the information
# necessary to rounding to any other rounding mode with p bits.  See the
# _divide_nearest function below for a nice example of this in practice.
#
# Here are primitives for integer division and shifting using round-to-odd.


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


def _digits_from_rational(a, b, closed=True):
    """
    Generate successive decimal digits for a fraction a / b in [0, 1].

    If closed is True (the default), the number x created from the generated
    digits is always largest s.t. x <= a / b.  If False, it's the largest
    such that x < a / b.

    a / b should be in the range [0, 1) if closed is True, and should be
    in (0, 1] if closed is False.

    >>> from itertools import islice
    >>> digits = _digits_from_rational(1, 7)
    >>> list(islice(digits, 20))
    [1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 1, 4]
    >>> digits = _digits_from_rational(3, 5)
    >>> list(islice(digits, 10))   #  3 / 5 = 0.600000.....
    [6, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> digits = _digits_from_rational(3, 5, closed=False)
    >>> list(islice(digits, 10))   #  3 / 5 = 0.599999.....
    [5, 9, 9, 9, 9, 9, 9, 9, 9, 9]

    """
    if not closed:
        a = b - a

    while True:
        digit, a = divmod(10 * a, b)
        yield digit if closed else 9 - digit


# Rounding directions.

class RoundingDirection(object):
    def __init__(self, _rounder):
        self._rounder = _rounder


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


# Attributes.

# XXX Default handler should set flag.
_attributes = {
    'rounding_direction': round_ties_to_even,
    'inexact_handler': lambda x: None,
    'invalid_operation_handler': lambda x: None,
}


def _current_rounding_direction():
    return _attributes['rounding_direction']


# Context managers to set and restore particular attributes.

@_contextlib.contextmanager
def rounding_direction(new_rounding_direction):
    old_rounding_direction = _attributes.get('rounding_direction')
    _attributes['rounding_direction'] = new_rounding_direction
    try:
        yield
    finally:
        _attributes['rounding_direction'] = old_rounding_direction


@_contextlib.contextmanager
def inexact_handler(new_inexact_handler):
    old_inexact_handler = _attributes.get('inexact_handler')
    _attributes['inexact_handler'] = new_inexact_handler
    try:
        yield
    finally:
        _attributes['inexact_handler'] = old_inexact_handler

@_contextlib.contextmanager
def invalid_operation_handler(new_invalid_operation_handler):
    old_invalid_operation_handler = _attributes.get('invalid_operation_handler')
    _attributes['invalid_operation_handler'] = new_invalid_operation_handler
    try:
        yield
    finally:
        _attributes['invalid_operation_handler'] = old_invalid_operation_handler


# Functions to signal various exceptions.  (Private for now, since only the
# core code needs to be able to signal exceptions.)

def _signal_inexact():
    _attributes['inexact_handler'](None)

def _signal_invalid_operation():
    _attributes['invalid_operation_handler'](None)



# Flags class.
#
# The internal routines here expect to be able to write to specific Flags
# attributes, but won't do any reading from them.

class _Flags(object):
    pass


class _NullFlags(object):
    """
    A NullFlags object simply ignores all writes to flag attributes, and
    raises an AttributeError on read.

    """
    # Catch writes to other attributes.
    __slots__ = ()

    def _set_error(self, error):
        pass

    def _get_error(self):
        raise AttributeError("'_NullFlags' object has no attribute 'error'")

    error = property(_get_error, _set_error)


_null_flags = _NullFlags()


class BinaryInterchangeFormat(object):
    """
    A BinaryInterchangeFormat instance represents one of the binary interchange
    formats described by IEEE 754-2008.  For example, the usual
    double-precision binary floating-point type is given by
    BinaryInterchangeFormat(width=64):

    >>> float64 = BinaryInterchangeFormat(width=64)

    Objects of this class should be treated as immutable.

    There are various attributes and read-only properties providing information
    about the format:

    >>> float64.precision  # precision in bits
    53
    >>> float64.width  # total width in bits
    64
    >>> float64.emax  # maximum exponent
    1023
    >>> float64.emin  # minimum exponent for normal numbers
    -1022

    Objects of this type are callable, and when called act like a class
    constructor to create floating-point numbers for the given format.

    >>> float64('2.3')
    BinaryInterchangeFormat(width=64)('2.3')
    >>> str(float64('2.3'))
    '2.3'

    """
    _class__cache = {}

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
    def class_(self):
        if self not in BinaryInterchangeFormat._class__cache:
            class BinaryFormat(_BinaryFloatBase):
                _format = self
            BinaryFormat.__name__ = 'Float{}'.format(self.width)
            BinaryInterchangeFormat._class__cache[self] = BinaryFormat

        return BinaryInterchangeFormat._class__cache[self]

    def _from_value(self, value=0):
        """
        Float<nnn>([value])

        Create a new Float<nnn> instance from the given input.

        """
        if isinstance(value, _BinaryFloatBase):
            # Initialize from another _BinaryFloatBase instance.
            return self._from_binary_float_base(value)

        elif isinstance(value, float):
            # Initialize from a float.
            return self._from_float(value)

        elif isinstance(value, _INTEGER_TYPES):
            # Initialize from an integer.
            return self._from_int(value)

        elif isinstance(value, _STRING_TYPES):
            # Initialize from a string.
            return self._from_str(value)

        else:
            raise TypeError(
                "Cannot construct a Float<nnn> instance from a "
                "value of type {}".format(type(value))
            )

    def _from_binary_float_base(self, b, flags=_null_flags):
        """
        Convert another _BinaryFloatBase instance to this format.

        """
        if b._type == _NAN:
            converted = self._from_nan_triple(
                sign=b._sign,
                signaling=b._signaling,
                payload=b._payload,
            )
        elif b._type == _INFINITE:
            # Infinities convert with no loss of information.
            converted = self._infinity(
                sign=b._sign,
            )
            flags.error = 0
        else:
            # Finite value.
            converted = self._from_triple(
                sign=b._sign,
                exponent=b._exponent,
                significand=b._significand,
                flags=flags,
            )
        return converted

    def _from_int(self, n, flags=_null_flags):
        """
        Convert the integer `n` to this format.

        """
        if n == 0:
            converted = self._zero(False)
            flags.error = 0
        else:
            sign, n = n < 0, abs(n)

            # Find d such that 2 ** (d - 1) <= n < 2 ** d.
            d = n.bit_length()

            exponent = max(d - self.precision, self.qmin) - 2
            significand = _rshift_to_odd(n, exponent)
            converted = self._final_round(sign, exponent, significand, flags)

        return converted

    def _from_str(self, s):
        """
        Convert an input string to this format.

        """
        m = _number_parser(s)
        if m is None:
            raise ValueError('invalid numeric string')

        sign = m.group('sign') == '-'

        if m.group('finite'):
            # Finite number.
            fraction = m.group('frac') or ''
            intpart = int(m.group('int') + fraction)
            exp = int(m.group('exp') or '0') - len(fraction)

            # Quick return for zeros.
            if not intpart:
                return self._zero(sign)

            # Express (absolute value of) incoming string in form a / b;
            # find d such that 2 ** (d - 1) <= a / b < 2 ** d.
            a, b = intpart * 10 ** max(exp, 0), 10 ** max(0, -exp)
            d = a.bit_length() - b.bit_length()
            d += (a >> d if d >= 0 else a << -d) >= b

            # Approximate a / b by number of the form q * 2 ** e.  We compute
            # two extra bits (hence the '- 2' below) of the result and round to
            # odd.
            exponent = max(d - self.precision, self.qmin) - 2
            significand = _divide_to_odd(
                a << max(-exponent, 0),
                b << max(exponent, 0),
            )
            return self._final_round(sign, exponent, significand)

        elif m.group('infinite'):
            # Infinity.
            return self._infinity(sign)

        elif m.group('nan'):
            # NaN.
            signaling = bool(m.group('signaling'))
            payload = int(m.group('payload') or 0)
            return self._from_nan_triple(
                sign=sign,
                signaling=signaling,
                payload=payload,
            )

        else:
            assert False, "Shouldn't get here."

    def _from_float(self, value, flags=_null_flags):
        """
        Convert a float to this format.

        """
        sign = _math.copysign(1.0, value) < 0

        if _math.isnan(value):
            # XXX Consider trying to extract and transfer the payload here.
            # We don't set the error flag for NaNs; maybe we should.
            converted = self._nan(
                sign=sign,
                signaling=False,
                payload=0,
            )

        elif _math.isinf(value):
            # Infinities.
            flags.error = 0
            converted = self._infinity(sign)
        elif value == 0.0:
            # Zeros
            flags.error = 0
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
            exponent = max(d - self.precision, self.qmin) - 2
            significand = _divide_to_odd(
                a << max(-exponent, 0),
                b << max(exponent, 0),
            )
            converted = self._final_round(sign, exponent, significand, flags)

        return converted

    def _from_nan(self, source):
        """
        Convert a NaN (possibly in a different format) to this format.

        Silently truncates the payload to fit when necessary.  Also converts a
        signaling NaN to a quiet NaN.

        """
        return self._from_nan_triple(
            sign=source._sign,
            signaling=False,
            payload=source._payload,
        )

    def _final_round(self, sign, e, q, flags=_null_flags):
        """
        Make final rounding adjustment, using the rounding mode from the
        current context.  For now, only round-ties-to-even is supported.

        """
        # Do the round ties to even, get rid of the 2 excess rounding bits.
        adj = _round_ties_to_even_offsets[q & 7]
        q, e = (q + adj) >> 2, e + 2

        # Check whether we need to adjust the exponent.
        if q.bit_length() == self.precision + 1:
            q, e = q >> 1, e + 1

        if e > self.qmax:
            flags.error = -1 if sign else 1
            return self._handle_overflow(sign)

        else:
            if sign:
                flags.error = (adj < 0) - (adj > 0)
            else:
                flags.error = (adj > 0) - (adj < 0)

            return self._finite(
                sign=sign,
                exponent=e,
                significand=q,
            )

    def _from_triple(self, sign, exponent, significand, flags=_null_flags):
        """
        Round the value (-1) ** sign * significand * 2 ** exponent to the
        format 'self'.

        """
        if significand == 0:
            flags.error = 0
            return self._zero(sign)

        d = exponent + significand.bit_length()

        # Find q such that q * 2 ** e approximates significand * 2 ** exponent.
        # Allow two extra bits for the final round.
        e = max(d - self.precision, self.qmin) - 2
        q = _rshift_to_odd(significand, e - exponent)
        return self._final_round(sign, e, q, flags=flags)

    def _handle_nans(self, *sources):
        # Look for signaling NaNs.
        for source in sources:
            if source._type == _NAN and source._signaling:
                _signal_invalid_operation()
                return self._from_nan(source)

        # All operands are quiet NaNs; return a result based on the first of
        # these.
        for source in sources:
            if source._type == _NAN:
                return self._from_nan(source)

        # If we get here, then _handle_nans has been called with all arguments
        # non-NaN.  This shouldn't happen.
        raise ValueError("_handle_nans didn't receive any NaNs.")

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
                return self._infinity(source1._sign)

        if source2._type == _INFINITE:
            return self._infinity(source2._sign)

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
        )

    def subtraction(self, source1, source2):
        """
        Return 'source1 - source2', rounded to the format given by 'self'.

        """
        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        # For non-NaNs, subtraction(a, b) is equivalent to
        # addition(a, b.negate())
        return self.addition(source1, source2.negate())

    def multiplication(self, source1, source2):
        """
        Return 'source1 * source2', rounded to the format given by 'self'.

        """
        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        sign = source1._sign ^ source2._sign
        if source1._type == _INFINITE:
            if source2.is_zero():
                return self._handle_invalid()
            else:
                return self._infinity(sign=sign)

        if source2._type == _INFINITE:
            if source1.is_zero():
                return self._handle_invalid()
            else:
                return self._infinity(sign=sign)

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
        if source1._type == _INFINITE:
            if source2._type == _INFINITE:
                return self._handle_invalid()
            else:
                return self._infinity(sign=sign)

        if source2._type == _INFINITE:
            # Already handled the case where source1 is infinite.
            return self._zero(sign=sign)

        if source1.is_zero():
            if source2.is_zero():
                return self._handle_invalid()
            else:
                return self._zero(sign=sign)

        if source2.is_zero():
            return self._infinity(sign=sign)

        # Finite / finite case.

        # First find d such that 2 ** (d-1) <= abs(source1) / abs(source2) <
        # 2 ** d.
        a = source1._significand
        b = source2._significand
        d = a.bit_length() - b.bit_length()
        d += (a >> d if d >= 0 else a << -d) >= b
        d += source1._exponent - source2._exponent

        # Exponent of result.  Reduce by 2 in order to compute a couple of
        # extra bits for rounding purposes.
        e = max(d - self.precision, self.qmin) - 2

        # Round (source1 / source2) * 2 ** -e to nearest integer.  source1 /
        # source2 * 2 ** -e == source1._significand / source2._significand *
        # 2 ** shift, where...
        shift = source1._exponent - source2._exponent - e

        a, b = a << max(shift, 0), b << max(0, -shift)
        q, r = divmod(a, b)
        # Round-to-odd.
        q |= bool(r)

        # Now result approximated by (-1) ** sign * q * 2 ** e.
        return self._final_round(sign, e, q)

    def square_root(self, source1):
        """
        Return the square root of source1 in format 'self'.

        """
        if source1._type == _NAN:
            return self._handle_nans(source1)

        # sqrt(+-0) is +-0.
        if source1.is_zero():
            return self._zero(sign=source1._sign)

        # Any nonzero negative number is invalid.
        if source1._sign:
            return self._handle_invalid()

        # sqrt(infinity) -> infinity.
        if source1._type == _INFINITE and not source1._sign:
            return self._infinity(sign=False)

        sig = source1._significand
        exponent = source1._exponent

        # Exponent of result.
        d = (sig.bit_length() + exponent + 1) // 2
        e = max(d - self.precision, self.qmin) - 2

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
        if (source1._type == _NAN or source2._type == _NAN or
            source3._type == _NAN):
            return self._handle_nans(source1, source2, source3)

        sign12 = source1._sign ^ source2._sign

        # Deal with infinities in the first two arguments.
        if source1._type == _INFINITE:
            if source2.is_zero():
                return self._handle_invalid()
            else:
                return self.addition(self._infinity(sign12), source3)

        if source2._type == _INFINITE:
            if source1.is_zero():
                return self._handle_invalid()
            else:
                return self.addition(self._infinity(sign12), source3)

        # Deal with zeros in the first two arguments.
        if source1.is_zero() or source2.is_zero():
            return self.addition(self._zero(sign12), source3)

        # Infinite 3rd argument.
        if source3._type == _INFINITE:
            return self._infinity(source3._sign)

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
        )

    def convert_from_int(self, n):
        """
        Convert the integer n to this format.

        """
        return self._from_int(n)

    def _zero(self, sign):
        """
        Return a suitably-signed zero for this format.

        """
        return self.class_(
            type=_FINITE,
            sign=sign,
            exponent=self.qmin,
            significand=0,
        )

    def _infinity(self, sign):
        """
        Return a suitably-signed infinity for this format.

        """
        return self.class_(
            type=_INFINITE,
            sign=sign,
        )

    def _from_nan_triple(self, sign, signaling, payload):
        """
        Return a NaN for this format, clipping the payload as appropriate.

        """
        min_payload = 1 if signaling else 0
        max_payload = (1 << self.precision - 2) - 1
        if payload < min_payload:
            payload = min_payload
        elif payload > max_payload:
            payload = max_payload
        return self._nan(
            sign=sign,
            signaling=signaling,
            payload=payload,
        )

    def _nan(self, sign, signaling, payload):
        """
        Return a NaN for this format.

        """
        return self.class_(
            type=_NAN,
            sign=sign,
            signaling=signaling,
            payload=payload,
        )

    def _finite(self, sign, exponent, significand):
        """
        Return a finite number in this format.

        """
        return self.class_(
            type=_FINITE,
            sign=sign,
            exponent=exponent,
            significand=significand,
        )

    def _common_format(fmt1, fmt2):
        """
        Return the common BinaryInterchangeFormat suitable for mixed binary
        operations with operands of types 'fmt1' and 'fmt2'.

        fmt1 and fmt2 should be instances of BinaryInterchangeFormat.

        """
        return fmt1 if fmt1.width >= fmt2.width else fmt2

    def _handle_overflow(self, sign):
        """
        Handle an overflow.

        """
        # For now, just returns the appropriate infinity.  Someday this should
        # handle rounding modes, flags, etc.
        return self._infinity(sign)

    def _handle_invalid(self, snan=None):
        """
        Handle an invalid operation.

        """
        if snan is not None:
            # Invalid operation from an snan: quiet the sNaN.
            return self._nan(
                sign=snan._sign,
                payload=snan._payload,
                signaling=False,
            )

        # For now, just return a quiet NaN.  Someday this should be more
        # sophisticated.
        return self._nan(
            sign=False,
            payload=0,
            signaling=False,
        )

    def _encode_as_int(self, source):
        """
        Encode 'source', which should have format 'self', as an unsigned int.

        """
        # Should only be used when 'source' has format 'self'.
        if not source._format == self:
            raise ValueError("Bad format.")

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
                return self._infinity(sign=sign)
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


_float64 = BinaryInterchangeFormat(64)


class _BinaryFloatBase(object):
    def __new__(cls, **kwargs):
        self = object.__new__(cls)
        self._type = kwargs.pop('type')
        self._sign = bool(kwargs.pop('sign'))

        if self._type == _FINITE:
            exponent = kwargs.pop('exponent')
            significand = kwargs.pop('significand')

            if not cls._format.qmin <= exponent <= cls._format.qmax:
                raise ValueError("exponent {} out of range".format(exponent))
            if not 0 <= significand < 1 << cls._format.precision:
                raise ValueError("significand out of range")

            # Check normalization.
            normalized = (
                significand >= 1 << cls._format.precision - 1 or
                exponent == cls._format.qmin
            )
            if not normalized:
                raise ValueError(
                    "Unnormalized significand ({}) or exponent ({}) "
                    "for {}.".format(
                        significand,
                        exponent,
                        cls._format,
                    )
                )

            self._exponent = int(exponent)
            self._significand = int(significand)
        elif self._type == _INFINITE:
            pass
        elif self._type == _NAN:
            payload = kwargs.pop('payload')
            signaling = kwargs.pop('signaling')

            # Payload must be at least 1 for a signaling nan, to avoid
            # confusion with the bit pattern for an infinity.
            min_payload = 1 if signaling else 0
            if not min_payload <= payload < 1 << (cls._format.precision - 2):
                raise ValueError("NaN payload out of range.")

            self._payload = int(payload)
            self._signaling = bool(signaling)

        else:
            raise ValueError("Unrecognized type: {}".format(self._type))

        return self

    def _equivalent(self, other):
        """
        Private method to determine whether self and other have the same
        structure.  Note that this is a much stronger check than equality: for
        example, -0.0 and 0.0 do not have the same structure; NaNs with
        different payloads are considered inequivalent (but those with the same
        payload are considered equivalent).

        Used only in testing.

        XXX: could replace this with a comparison of corresponding byte
        strings.

        """
        if self._type == other._type == _FINITE:
            return (
                self._sign == other._sign and
                self._exponent == other._exponent and
                self._significand == other._significand
            )
        elif self._type == other._type == _INFINITE:
            return (
                self._sign == other._sign
            )
        elif self._type == other._type == _NAN:
            return (
                self._sign == other._sign and
                self._payload == other._payload and
                self._signaling == other._signaling
            )
        else:
            return False

    def _to_short_str(self):
        """
        Convert to a shortest Decimal string that rounds back to the given
        value.

        """
        # Quick returns for zeros, infinities, NaNs.
        if self._type == _FINITE and self._significand == 0:
            return '-0.0' if self._sign else '0.0'

        if self._type == _INFINITE:
            return '-Infinity' if self._sign else 'Infinity'

        if self._type == _NAN:
            return '{sign}{signaling}NaN({payload})'.format(
                sign='-' if self._sign else '',
                signaling='s' if self._signaling else '',
                payload=self._payload,
            )

        # General nonzero finite case.

        # Interval of values that round to self is
        # (high / denominator, low / denominator)

        # Is this a power of 2 that falls between two *normal* binades (so
        # that the ulp function has a discontinuity at this point)?
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

        # Find appropriate power of 10.
        # Invariant: 10 ** (n-1) <= high / denominator < 10 ** n.
        n = len(str(high)) - len(str(denominator))
        n += (high // 10 ** n if n >= 0 else high * 10 ** -n) >= denominator

        # So now we want to compute digits of high / denominator * 10 ** -n.
        high *= 10 ** max(-n, 0)
        target *= 10 ** max(-n, 0)
        low *= 10 ** max(-n, 0)
        denominator *= 10 ** max(0, n)

        assert 0 < low < target < high < denominator

        # The interval of values that will round back to self is closed if
        # the significand is even, and open otherwise.
        closed = self._significand % 2 == 0

        high_digits = _digits_from_rational(high, denominator, closed=closed)
        low_digits = _digits_from_rational(low, denominator, closed=not closed)
        pairs = _zip(low_digits, high_digits)

        digits = []
        for low_digit, high_digit in pairs:
            if low_digit != high_digit:
                break
            digits.append(high_digit)
            target = 10 * target - high_digit * denominator

        # The best final digit is the digit giving the closest decimal string
        # to the target value amongst all digits in (low_digit, high_digit].
        # In most cases this just means the closest digit; the exception occurs
        # when self is a power of 2, so that the interval of values rounding to
        # self isn't centered on self; in that case, the nearest digit may lie
        # below the interval (low_digit, high_digit].
        best_final_digit = _divide_nearest(10 * target, denominator)
        best_final_digit = max(best_final_digit, low_digit + 1)

        assert low_digit < best_final_digit <= high_digit
        digits.append(best_final_digit)

        if digits == [1]:
            # Special corner case: it's possible in this case that the
            # actual closest short string to the target starts with 0.
            # Recompute.
            best_final_digit2 = _divide_nearest(100 * target, denominator)
            if best_final_digit2 < 10:
                # No need to check that this is in range, since this special
                # case can only occur for subnormals, and there the original
                # interval is always symmetric.
                digits = [0, best_final_digit2]

        # Cheat by getting the decimal module to do the string formatting
        # (insertion of decimal point, etc.) for us.
        return str(
            _decimal.Decimal(
                '{0}0.{1}e{2}'.format(
                    '-' if self._sign else '',
                    ''.join(map(str, digits)),
                    n,
                 )
            )
        )

    def __repr__(self):
        return "{!r}({!r})".format(self._format, self._to_short_str())

    def __str__(self):
        return self._to_short_str()

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
        if not quiet and q << 2 != to_quarter:
            _signal_inexact()

        # Normalize.
        if q == 0:
            return self._format._zero(self._sign)
        else:
            shift = self._format.precision - q.bit_length()
            return self._format._finite(self._sign, -shift, q << shift)

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
        if not other._format == self._format:
            raise ValueError("remainder args should be of the same format")

        # NaNs follow the usual rules.
        if self._type == _NAN or other._type == _NAN:
            return self._format._handle_nans(self, other)

        # remainder(infinity, y) and remainder(x, 0) are invalid
        if self._type == _INFINITE or other.is_zero():
            _signal_invalid_operation()
            # Return the standard NaN.
            return self._format._nan(False, False, 0)

        # remainder(x, +/-infinity) is x for any finite x.  Similarly, if x is
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
        e = max(exponent + significand.bit_length() - self._format.precision, self._format.qmin)
        return self._format._finite(sign, e, significand << exponent - e)

    def min_num(self, other):
        """
        Minimum of self and other.

        If self and other are numerically equal (for example in the case of
        differently-signed zeros), self is returned.

        """
        # This is a homogeneous operation: both operands have the same format.
        if not other._format == self._format:
            raise ValueError("remainder args should be of the same format")

        # Special behaviour for NaNs: if one operand is NaN and the other is not
        # return the non-NaN operand.
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
        if not other._format == self._format:
            raise ValueError("remainder args should be of the same format")

        # Special behaviour for NaNs: if one operand is NaN and the other is not
        # return the non-NaN operand.
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
        if not other._format == self._format:
            raise ValueError("remainder args should be of the same format")

        # Special behaviour for NaNs: if one operand is NaN and the other is not
        # return the non-NaN operand.
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
        if not other._format == self._format:
            raise ValueError("remainder args should be of the same format")

        # Special behaviour for NaNs: if one operand is NaN and the other is not
        # return the non-NaN operand.
        if self.is_nan() and not self.is_signaling() and not other.is_nan():
            return other
        if other.is_nan() and not other.is_signaling() and not self.is_nan():
            return self

        # Apart from the above special case, treat NaNs as normal.
        if self._type == _NAN or other._type == _NAN:
            return self._format._handle_nans(self, other)

        cmp = _compare_ordered(self.abs(), other.abs())
        return other if cmp <= 0 else self

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
            return type(self)(
                type=_FINITE,
                sign=self._sign,
                exponent=self._exponent,
                significand=self._significand,
            )

        elif self._type == _INFINITE:
            return type(self)(
                type=_INFINITE,
                sign=self._sign,
            )

        elif self._type == _NAN:
            return type(self)(
                type=_NAN,
                sign=self._sign,
                payload=self._payload,
                signaling=self._signaling,
            )

        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

    def negate(self):
        """
        Return the negation of self.

        """
        if self._type == _FINITE:
            return type(self)(
                type=_FINITE,
                sign=not self._sign,
                exponent=self._exponent,
                significand=self._significand,
            )

        elif self._type == _INFINITE:
            return type(self)(
                type=_INFINITE,
                sign=not self._sign,
            )

        elif self._type == _NAN:
            return type(self)(
                type=_NAN,
                sign=not self._sign,
                payload=self._payload,
                signaling=self._signaling,
            )

        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

    def abs(self):
        """
        Return the absolute value of self.

        """
        if self._type == _FINITE:
            return type(self)(
                type=_FINITE,
                sign=False,
                exponent=self._exponent,
                significand=self._significand,
            )
        elif self._type == _INFINITE:
            return type(self)(
                type=_INFINITE,
                sign=False,
            )
        elif self._type == _NAN:
            return type(self)(
                type=_NAN,
                sign=False,
                payload=self._payload,
                signaling=self._signaling,
            )
        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

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
            return type(self)(
                type=_FINITE,
                sign=other._sign,
                exponent=self._exponent,
                significand=self._significand,
            )
        elif self._type == _INFINITE:
            return type(self)(
                type=_INFINITE,
                sign=other._sign,
            )
        elif self._type == _NAN:
            return type(self)(
                type=_NAN,
                sign=other._sign,
                payload=self._payload,
                signaling=self._signaling,
            )
        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

    def __pos__(self):
        return self.negate()

    def __neg__(self):
        return self.negate()

    def __abs__(self):
        return self.abs()

    def _convert_other(self, other):
        """
        Given numeric operands self and other, with self an instance of
        _BinaryFloatBase, convert other to an operand of type _BinaryFloatBase
        if necessary, and return the common format for the output.

        Return a pair converted_other, common_format

        """
        # Convert other.
        if isinstance(other, _BinaryFloatBase):
            pass
        elif isinstance(other, float):
            other = _float64._from_float(other)
        elif isinstance(other, _INTEGER_TYPES):
            other = self._format._from_int(other)
        else:
            raise TypeError(
                "Can't convert operand {} of type {} to "
                "_BinaryFloatBase.".format(
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

    if _sys.version_info.major == 2:
        # Make sure that Python 2 divisions involving these types behave the
        # same way regardless of whether the division __future__ import is in
        # effect or not.
        __div__ = __truediv__
        __rdiv__ = __rtruediv__

    # Overloaded comparisons.

    def _rich_compare_general(self, other, operator, unordered_result):
        # operator should be one of the 6 comparison operators from
        # the operator module.

        flags = _Flags()
        if isinstance(other, _INTEGER_TYPES):
            other = self._format._from_int(other, flags)

        elif isinstance(other, float):
            other = self._format._from_float(other, flags)

        elif isinstance(other, _BinaryFloatBase):
            other = self._format._from_binary_float_base(other, flags)

        else:
            raise NotImplementedError

        if self.is_signaling() or other.is_signaling():
            return _handle_invalid_bool(unordered_result)
        elif self._type == _NAN or other._type == _NAN:
            return unordered_result
        else:
            result = _compare_ordered(self, other) or flags.error
            return operator(result, 0)

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
        elif _sys.version_info.major == 2:
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
            # Assuming Python >= 3.2, compatibility with floats and ints
            # (not to mention Fraction and Decimal instances) follows if
            # we use the formulas described in the Python docs.
            if self._exponent >= 0:
                exp_hash = pow(2, self._exponent, _PyHASH_MODULUS)
            else:
                exp_hash = pow(_PyHASH_2INV, -self._exponent, _PyHASH_MODULUS)
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
    # This function should only ever be called for two BinaryFloatBase
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
    raise ValueError("Comparison involving signaling NaN")


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
        flags = _Flags()
        source2 = source1._format._from_binary_float_base(source2, flags)
        result = _compare_ordered(source1, source2) or flags.error
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
        flags = _Flags()
        source2 = source1._format._from_binary_float_base(source2, flags)
        result = _compare_ordered(source1, source2) or flags.error
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
