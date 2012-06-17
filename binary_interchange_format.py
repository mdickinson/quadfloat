from __future__ import division

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
    from future_builtins import zip

    # Values used to compute hashes.
    if _sys.maxint == 2 ** 31 - 1:
        _PyHASH_MODULUS = 2 ** 31 - 1
    elif _sys.maxint == 2 ** 63 - 1:
        _PyHASH_MODULUS == 2 ** 61 - 1
    _PyHASH_2INV = pow(2, _PyHASH_MODULUS - 2, _PyHASH_MODULUS)
    _PyHASH_INF = hash(float('inf'))
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
_round_toward_zero_offsets = [0, -1, -2, -3, 0, -1, -2, -3]
_round_ties_to_away_offsets = [0, -1, 2, 1, 0, -1, 2, 1]


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


def _divide_to_odd(a, b):
    """
    Compute a / b, rounding inexact results to the nearest *odd*
    integer.

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
    Compute a / 2**shift, rounding inexact results to the nearest *odd*
    integer.

    """
    if shift <= 0:
        return a << -shift
    else:
        return (a >> shift) | bool(a & ~(-1 << shift))


def _divide_nearest(a, b):
    """
    Compute the nearest integer to the quotient a / b, rounding ties to the
    nearest even integer.  a and b should be integers, with b positive.

    """
    q = _divide_to_odd(4 * a, b)
    return (q + _round_ties_to_even_offsets[q & 7]) >> 2


def _digits_from_rational(a, b, closed=True):
    """
    Generate successive decimal digits for a fraction a / b in [0, 1].

    If closed is True (the default), the number x created from the generated
    digits is always largest s.t. x <= a / b.  If False, it's the largest
    such that x < a / b.

    """
    if closed:
        if not 0 <= a < b:
            raise ValueError(
                "a and b should satisfy 0 <= a < b in _digits_from_rational"
            )
    else:
        if not 0 < a <= b:
            raise ValueError(
                "a and b should satisfy 0 < a <= b in _digits_from_rational"
            )

    if not closed:
        a = b - a

    while True:
        digit, a = divmod(10 * a, b)
        yield digit if closed else 9 - digit


_Flags = dict


def _handle_invalid_bool(default_bool):
    raise ValueError("Comparison involving signaling NaN")


class BinaryInterchangeFormat(object):
    _class__cache = {}

    def __new__(cls, width):
        valid_width = width in {16, 32, 64} or width >= 128 and width % 32 == 0
        if not valid_width:
            raise ValueError(
                "For an interchange format, width should be 16, 32, 64, "
                "or a multiple of 32 that's greater than 128."
            )
        self = object.__new__(cls)
        self.width = width
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
    def qbias(self):
        return 1 - self.qmin

    @property
    def _exponent_field_width(self):
        """
        Number of bits used to encode exponent for an interchange format.

        """
        return self.width - self.precision

    @property
    def _decimal_places(self):
        """
        Minimal number of digits necessary to provide roundtrip conversions.

        """
        # Formula: 1 + ceiling(self.precision * log10(2)) or equivalently, 2 +
        #  floor(self.precision * log10(2)), (since precision >= 1 and log10(2)
        #  is irrational).
        return len(str(1 << self.precision)) + 1

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
        if isinstance(value, float):
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

    def _from_int(self, n, flags=None):
        """
        Convert the integer `n` to this format.

        """
        if n == 0:
            converted = self._zero(False)
            if flags is not None:
                flags['error'] = 0
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

            # Parse payload, and clip to bounds if necessary.
            payload = int(m.group('payload') or 0)
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
        else:
            assert False, "Shouldn't get here."

    def _from_float(self, value, flags=None):
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
            if flags is not None:
                flags['error'] = 0
            converted = self._infinity(sign)
        elif value == 0.0:
            # Zeros
            if flags is not None:
                flags['error'] = 0
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
        max_payload = (1 << (self.precision - 2)) - 1

        return self._nan(
            sign=source._sign,
            signaling=False,
            payload=min(source._payload, max_payload),
        )

    def _final_round(self, sign, e, q, flags=None):
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
            if flags is not None:
                flags['error'] = -1 if sign else 1
            return self._handle_overflow(sign)

        else:
            if flags is not None:
                if sign:
                    flags['error'] = (adj < 0) - (adj > 0)
                else:
                    flags['error'] = (adj > 0) - (adj < 0)

            return self._finite(
                sign=sign,
                exponent=e,
                significand=q,
            )

    def _from_triple(self, sign, exponent, significand):
        """
        Round the value (-1) ** sign * significand * 2 ** exponent to the
        format 'self'.

        """
        if significand == 0:
            return self._zero(sign)

        d = exponent + significand.bit_length()

        # Find q such that q * 2 ** e approximates significand * 2 ** exponent.
        # Allow two extra bits for the final round.
        e = max(d - self.precision, self.qmin) - 2
        q = _rshift_to_odd(significand, e - exponent)
        return self._final_round(sign, e, q)

    def _handle_nans(self, *sources):
        # Look for signaling NaNs.
        for source in sources:
            if source._type == _NAN and source._signaling:
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

    def decode(self, encoded_value):
        """
        Decode a string of bytes to the corresponding Float<nnn> instance.

        """
        if len(encoded_value) != self.width // 8:
            raise ValueError("Wrong number of bytes for format.")

        exponent_field_width = self._exponent_field_width
        significand_field_width = self.precision - 1

        # Extract fields.
        equivalent_int = _int_from_bytes(encoded_value)
        significand_field = (
            equivalent_int &
            ((1 << significand_field_width) - 1)
        )
        equivalent_int >>= significand_field_width
        exponent_field = equivalent_int & ((1 << exponent_field_width) - 1)
        equivalent_int >>= exponent_field_width
        sign = bool(equivalent_int)

        assert 0 <= exponent_field < (1 << exponent_field_width)
        assert 0 <= significand_field < (1 << significand_field_width)

        # Construct value.
        if exponent_field == (1 << exponent_field_width) - 1:
            # Infinities, Nans.
            if significand_field == 0:
                # Infinities.
                return self._infinity(sign=sign)
            else:
                # Nan.
                payload_width = significand_field_width - 1
                payload = significand_field & ((1 << payload_width) - 1)
                significand_field >>= payload_width
                # Top bit of significand field indicates whether this Nan is
                # quiet (1) or signaling (0).
                assert 0 <= significand_field <= 1
                signaling = not significand_field
                return self._nan(sign, signaling, payload)
        elif exponent_field == 0:
            # Subnormals, Zeros.
            return self._finite(
                sign=sign,
                exponent=self.qmin,
                significand=significand_field,
            )
        else:
            # Normal number.
            return self._finite(
                sign=sign,
                exponent=exponent_field - self.qbias,
                significand=significand_field + (1 << self.precision - 1),
            )


_Float64 = BinaryInterchangeFormat(64)


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

        # XXX Deal with the special case where there's a power of 10 in the
        # interval.

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
        pairs = zip(low_digits, high_digits)

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
        return "{}('{}')".format(type(self).__name__, self._to_short_str())

    def __str__(self):
        return self._to_short_str()

    # IEEE 5.7.2: General operations.

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
        Encode a Float<nnn> instance as a 16-character bytestring.

        """
        exponent_field_width = self._format._exponent_field_width
        significand_field_width = self._format.precision - 1

        if self._type == _FINITE:
            if self.is_subnormal() or self.is_zero():
                exponent_field = 0
                significand_field = self._significand
            else:
                exponent_field = self._exponent + self._format.qbias
                significand_field = (
                    self._significand - (1 << self._format.precision - 1)
                )
        elif self._type == _INFINITE:
            exponent_field = (1 << exponent_field_width) - 1
            significand_field = 0
        elif self._type == _NAN:
            exponent_field = (1 << exponent_field_width) - 1
            significand_field = (
                ((not self._signaling) << significand_field_width - 1) +
                self._payload
            )
        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

        equivalent_int = (
            (self._sign << (exponent_field_width + significand_field_width)) +
            (exponent_field << significand_field_width) +
            significand_field
        )

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
            other = _Float64._from_float(other)
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
            flags['error'] = 0

        else:
            raise NotImplementedError

        if self._type == _NAN or other._type == _NAN:
            return _compare_nans(self, other, unordered_result)
        result = _compare_ordered(self, other) or flags['error']
        return operator(result, 0)

    def _py27_hash(self):
        """
        Return hash value compatible with ints and floats.

        Raise TypeError for signaling NaNs.

        """
        if self._type == _NAN:
            if self._signaling:
                raise ValueError('Signaling NaNs are unhashable.')

            return hash(float('nan'))

        if self._type == _INFINITE:
            return hash(float('-inf')) if self._sign else hash(float('inf'))

        # Now we've got something finite.  If it's an integer, return
        # the hash of the corresponding integer.
        # XXX. This is needlessly inefficient for huge values.
        if self == int(self):
            return hash(int(self))

        # Otherwise, if it's equal to the corresponding Python float,
        # return the hash of that float.
        elif self == float(self):
            return hash(float(self))

        # Otherwise, don't worry about matching anything else, and just
        # return the py3k hash.
        else:
            return self._py32_hash()

    def _py32_hash(self):
        """
        Return hash value compatible with other numeric types.
        Uses the formulas described in the 'built-in types' section
        of the Python docs.

        """
        if self._type == _NAN:
            if self._signaling:
                raise ValueError('Signaling NaNs are unhashable.')
            return _PyHASH_NAN
        elif self._type == _INFINITE:
            return -_PyHASH_INF if self._sign else _PyHASH_INF
        else:
            if self._exponent >= 0:
                exp_hash = pow(2, self._exponent, _PyHASH_MODULUS)
            else:
                exp_hash = pow(_PyHASH_2INV, -self._exponent, _PyHASH_MODULUS)
            hash_ = self._significand * exp_hash % _PyHASH_MODULUS
            ans = -hash_ if self._sign else hash_

            return -2 if ans == -1 else ans

    if _sys.version_info.major == 2:
        def __hash__(self):
            return self._py27_hash()
    else:
        def __hash__(self):
            return self._py32_hash()

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
    def convert_to_integer_ties_to_even(self):
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
        q = (q + _round_ties_to_even_offsets[q & 7]) >> 2
        # Use int() to convert from long if necessary
        return int(-q if self._sign else q)

    def convert_to_integer_toward_zero(self):
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
        q = (q + _round_toward_zero_offsets[q & 7]) >> 2
        # Use int() to convert from long if necessary
        return int(-q if self._sign else q)

    def convert_to_integer_toward_positive(self):
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
        # int() to convert from long if necessary
        return int(-((q if self._sign else -q) >> 2))

    def convert_to_integer_toward_negative(self):
        if self._type == _NAN:
            # XXX Signaling nans should also raise the invalid operation
            # exception.
            raise ValueError("Cannot convert a NaN to an integer.")

        if self._type == _INFINITE:
            # NB. Python raises OverflowError here, which doesn't really
            # seem right.
            raise ValueError("Cannot convert an infinity to an integer.")

        # (-1) ** sign * significand * 2 ** exponent

        # Round to odd, with 2 extra bits.
        q = _rshift_to_odd(self._significand, -self._exponent - 2)
        # int() to convert from long if necessary
        return int((-q if self._sign else q) >> 2)

    def convert_to_integer_ties_to_away(self):
        if self._type == _NAN:
            # XXX Signaling nans should also raise the invalid operation
            # exception.
            raise ValueError("Cannot convert a NaN to an integer.")

        if self._type == _INFINITE:
            # NB. Python raises OverflowError here, which doesn't really
            # seem right.
            raise ValueError("Cannot convert an infinity to an integer.")

        # (-1) ** sign * significand * 2 ** exponent

        # Compute significand * 2 ** (exponent + 2), rounded to the nearest
        # integer using round-to-odd.  Extra bits will be used for rounding.
        q = _rshift_to_odd(self._significand, -self._exponent - 2)
        q = (q + _round_ties_to_away_offsets[q & 7]) >> 2
        # int() to convert from long if necessary
        return int(-q if self._sign else q)


# Section 5.6.1: Comparisons.

def _compare_ordered(source1, source2):
    """
    Given non-NaN values source1 and source2, compare them, returning -1, 0 or
    1 according as source1 < source2, source1 == source2, or source1 > source2.

    """
    # Zeros.
    if source1.is_zero():
        if source2.is_zero():
            # Two zeros are equal.
            return 0
        else:
            # cmp(0, x):  -1 if x is positive, 1 if x is negative
            return 1 if source2._sign else -1
    elif source2.is_zero():
        # cmp(x, 0)
        return -1 if source1._sign else 1

    # Different signs.
    if source1._sign != source2._sign:
        return -1 if source1._sign else 1

    # Infinities.
    if source1._type == _INFINITE:
        if source2._type == _INFINITE:
            # Must have the same signs, so they're equal
            return 0
        else:
            # cmp(inf, finite)
            return -1 if source1._sign else 1
    elif source2._type == _INFINITE:
        return 1 if source2._sign else -1

    # Compare adjusted exponents.
    source1_exp = source1._significand.bit_length() + source1._exponent
    source2_exp = source2._significand.bit_length() + source2._exponent
    if source1_exp != source2_exp:
        cmp = -1 if source1_exp < source2_exp else 1
        return -cmp if source1._sign else cmp

    # Compare significands.
    exponent_diff = source1._exponent - source2._exponent
    a = source1._significand << max(exponent_diff, 0)
    b = source2._significand << max(0, -exponent_diff)
    if a != b:
        cmp = -1 if a < b else 1
        return -cmp if source1._sign else cmp

    # Values are equal.
    return 0


def _compare_nans(source1, source2, quiet_nan_result):
    """
    Do a comparison in the case that either source1 or source2 is
    a NaN (quiet or signaling).

    nan_result is the result to return in nonstop mode.

    """
    if source1.is_signaling() or source2.is_signaling():
        return _handle_invalid_bool(quiet_nan_result)
    else:
        return quiet_nan_result


def compare_quiet_equal(source1, source2):
    """
    Return True if source1 and source2 are numerically equal, else False.

    """
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, False)
    return _compare_ordered(source1, source2) == 0


def compare_quiet_not_equal(source1, source2):
    """
    Return True if source1 and source2 are numerically equal, else False.

    """
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, True)
    return _compare_ordered(source1, source2) != 0


def compare_quiet_greater(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, False)
    return _compare_ordered(source1, source2) > 0


def compare_quiet_greater_equal(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, False)
    return _compare_ordered(source1, source2) >= 0


def compare_quiet_less(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, False)
    return _compare_ordered(source1, source2) < 0


def compare_quiet_less_equal(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, False)
    return _compare_ordered(source1, source2) <= 0


def compare_quiet_unordered(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, True)
    return False


def compare_quiet_not_greater(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, True)
    return _compare_ordered(source1, source2) <= 0


def compare_quiet_less_unordered(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, True)
    return _compare_ordered(source1, source2) < 0


def compare_quiet_not_less(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, True)
    return _compare_ordered(source1, source2) >= 0


def compare_quiet_greater_unordered(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, True)
    return _compare_ordered(source1, source2) > 0


def compare_quiet_ordered(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _compare_nans(source1, source2, False)
    return True


def compare_signaling_equal(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(False)
    return _compare_ordered(source1, source2) == 0


def compare_signaling_greater(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(False)
    return _compare_ordered(source1, source2) > 0


def compare_signaling_greater_equal(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(False)
    return _compare_ordered(source1, source2) >= 0


def compare_signaling_less(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(False)
    return _compare_ordered(source1, source2) < 0


def compare_signaling_less_equal(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(False)
    return _compare_ordered(source1, source2) <= 0


def compare_signaling_not_equal(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(True)
    return _compare_ordered(source1, source2) != 0


def compare_signaling_not_greater(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(True)
    return _compare_ordered(source1, source2) <= 0


def compare_signaling_less_unordered(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(True)
    return _compare_ordered(source1, source2) < 0


def compare_signaling_not_less(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(True)
    return _compare_ordered(source1, source2) >= 0


def compare_signaling_greater_unordered(source1, source2):
    if source1._type == _NAN or source2._type == _NAN:
        return _handle_invalid_bool(True)
    return _compare_ordered(source1, source2) > 0
