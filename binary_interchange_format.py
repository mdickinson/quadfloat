import decimal as _decimal
import math as _math
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

else:
    _STRING_TYPES = str,
    _INTEGER_TYPES = int,
    _int_to_bytes = lambda n, length: n.to_bytes(length, byteorder='little')
    _int_from_bytes = lambda bs: int.from_bytes(bs, byteorder='little')


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
        return self.class_._from_value(*args, **kwargs)

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
        return len(str(2 ** self.precision)) + 1

    @property
    def class_(self):
        if self not in BinaryInterchangeFormat._class__cache:
            class BinaryFormat(_BinaryFloatBase):
                _format = self
            BinaryFormat.__name__ = 'Float{}'.format(self.width)
            BinaryInterchangeFormat._class__cache[self] = BinaryFormat

        return BinaryInterchangeFormat._class__cache[self]

    def _from_nan(self, source):
        """
        Convert a NaN (possibly in a different format) to this format.

        Silently truncates the payload to fit when necessary.  Also converts a
        signaling NaN to a quiet NaN.

        """
        max_payload = 2 ** (self.precision - 2) - 1

        return self.class_(
            type=_NAN,
            sign=source._sign,
            signaling=False,
            payload=min(source._payload, max_payload),
        )

    def _final_round(self, sign, e, q):
        """
        Make final rounding adjustment, using the rounding mode from the
        current context.  For now, only round-half-to-even is supported.

        """
        # Do the round half to even, get rid of the 2 excess rounding bits.
        _round_half_to_even_offsets = [0, -1, -2, 1, 0, -1, 2, 1]
        q += _round_half_to_even_offsets[q & 7]
        q, e = q >> 2, e + 2

        # Check whether we need to adjust the exponent.
        if q.bit_length() == self.precision + 1:
            q >>= 1
            e += 1

        # Overflow.
        if e > self.qmax:
            return self._handle_overflow(sign)

        return self.class_(
            type=_FINITE,
            sign=sign,
            exponent=e,
            significand=q,
        )

    def _round_from_triple(self, sign, exponent, significand):
        """
        Round the value (-1)**sign * significand * 2**exponent to the format
        'self'.

        """
        if significand == 0:
            return self.class_(
                type=_FINITE,
                sign=sign,
                exponent=self.qmin,
                significand=0,
            )

        # ... first find exponent e of result.  Allow two extra bits for doing
        # later rounding.
        d = exponent + significand.bit_length()
        e = max(d - self.precision, self.qmin) - 2

        # Find q such that q * 2**e approximates significand * 2**exponent.
        shift = exponent - e
        if shift >= 0:
            q = significand << shift
        else:
            # round-to-odd
            q = (significand >> -shift) | bool(significand & ~(-1 << -shift))

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

        return self._round_from_triple(
            sign=sign,
            exponent=exponent,
            significand=abs(significand),
        )

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
        return self._round_from_triple(
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

        # First find d such that 2**(d-1) <= abs(source1) / abs(source2) <
        # 2**d.
        a = source1._significand
        b = source2._significand
        d = a.bit_length() - b.bit_length()
        d += (a >> d if d >= 0 else a << -d) >= b
        d += source1._exponent - source2._exponent

        # Exponent of result.  Reduce by 2 in order to compute a couple of
        # extra bits for rounding purposes.
        e = max(d - self.precision, self.qmin) - 2

        # Round (source1 / source2) * 2**-e to nearest integer.  source1 /
        # source2 * 2**-e == source1._significand / source2._significand *
        # 2**shift, where...
        shift = source1._exponent - source2._exponent - e

        a, b = a << max(shift, 0), b << max(0, -shift)
        q, r = divmod(a, b)
        # Round-to-odd.
        q |= bool(r)

        # Now result approximated by (-1)**sign * q * 2**e.
        return self._final_round(sign, e, q)

    def subtraction(self, source1, source2):
        """
        Return the difference source1 - source2.

        """
        if source1._type == _NAN or source2._type == _NAN:
            return self._handle_nans(source1, source2)

        # For non-NaNs, subtraction(a, b) is equivalent to
        # addition(a, b.negate())
        return self.addition(source1, source2.negate())

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

    def decode(self, encoded_value):
        return self.class_.decode(encoded_value)

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
        cls = self.class_
        return cls(type=_INFINITE, sign=sign)

    def _handle_invalid(self, snan=None):
        """
        Handle an invalid operation.

        """
        cls = self.class_
        if snan is not None:
            # Invalid operation from an snan: quiet the sNaN.
            return cls(
                type=_NAN,
                sign=snan._sign,
                payload=snan._payload,
                signaling=False,
            )

        # For now, just return a quiet NaN.  Someday this should be more
        # sophisticated.
        return cls(
            type=_NAN,
            sign=False,
        )


_Float64 = BinaryInterchangeFormat(64)


class _BinaryFloatBase(object):
    def __new__(cls, **kwargs):
        type = kwargs.pop('type')
        sign = kwargs.pop('sign')
        if type == _FINITE:
            exponent = kwargs.pop('exponent')
            significand = kwargs.pop('significand')

            if not cls._format.qmin <= exponent <= cls._format.qmax:
                raise ValueError("exponent {} out of range".format(exponent))
            if not 0 <= significand < 2 ** cls._format.precision:
                raise ValueError("significand out of range")

            # Check normalization.
            normalized = (
                significand >= 2 ** (cls._format.precision - 1) or
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

            self = object.__new__(cls)
            self._exponent = int(exponent)
            self._significand = int(significand)
        elif type == _INFINITE:
            self = object.__new__(cls)
        elif type == _NAN:
            # XXX Not sure why we're giving defaults here;  maybe we should
            # always specify on construction (and use nice defaults in suitable
            # helper functions).
            payload = kwargs.pop('payload', 0)
            signaling = kwargs.pop('signaling', False)

            # Payload must be at least 1 for a signaling nan, to avoid
            # confusion with the bit pattern for an infinity.
            min_payload = 1 if signaling else 0
            if not min_payload <= payload < 2 ** (cls._format.precision - 2):
                raise ValueError("NaN payload out of range.")

            self = object.__new__(cls)
            self._payload = int(payload)
            self._signaling = bool(signaling)

        else:
            raise ValueError("Unrecognized type: {}".format(type))

        self._type = type
        self._sign = bool(sign)
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

    @classmethod
    def _from_value(cls, value=0):
        """
        Float<nnn>([value])

        Create a new Float<nnn> instance from the given input.

        """
        if isinstance(value, float):
            # Initialize from a float.
            return cls._from_float(value)

        elif isinstance(value, _INTEGER_TYPES):
            # Initialize from an integer.
            return cls._from_int(value)

        elif isinstance(value, _STRING_TYPES):
            # Initialize from a string.
            return cls._from_str(value)

        else:
            raise TypeError(
                "Cannot construct a Float<nnn> instance from a "
                "value of type {}".format(type(value))
            )

    def _to_str(self, places=None):
        """
        Convert to a Decimal string with a given
        number of significant digits.

        """
        if self._type == _FINITE:
            if self._significand == 0:
                if self._sign:
                    return '-0.0'
                else:
                    return '0.0'

            if places is None:
                # Sufficient places to recover the value.
                places = self._format._decimal_places

            # Find a, b such that a / b = abs(self)
            a = self._significand << max(self._exponent, 0)
            b = 1 << max(0, -self._exponent)

            # Compute exponent m for result.
            n = len(str(a)) - len(str(b))
            n += (a // 10 ** n if n >= 0 else a * 10 ** -n) >= b
            # Invariant: 10 ** (n - 1) <= abs(self) < 10 ** n.
            m = n - places

            # Approximate a / b by a number of the form q * 10 ** m
            a, b = a * 10 ** max(-m, 0), b * 10 ** max(0, m)

            # Now divide to get quotient and remainder.
            q, r = divmod(a, b)
            assert 10 ** (places - 1) <= q < 10 ** places
            if 2 * r > b or 2 * r == b and q & 1:
                q += 1
                if q == 10 ** places:
                    q //= 10
                    m += 1

            # Cheat by getting the decimal module to do the string formatting
            # (insertion of decimal point, etc.) for us.
            return str(
                _decimal.Decimal(
                    '{0}{1}e{2}'.format(
                        '-' if self._sign else '',
                        q,
                        m,
                    )
                )
            )
        elif self._type == _INFINITE:
            return '-Infinity' if self._sign else 'Infinity'

        elif self._type == _NAN:
            pieces = []
            if self._sign:
                pieces.append('-')
            if self._signaling:
                pieces.append('s')
            pieces.append('NaN')
            pieces.append('({})'.format(self._payload))
            return ''.join(pieces)

        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

    def __repr__(self):
        return "{}('{}')".format(type(self).__name__, self._to_str())

    def __str__(self):
        return self._to_str()

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
            2 ** (self._format.precision - 1) <= self._significand
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
            0 < self._significand < 2 ** (self._format.precision - 1)
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

    @classmethod
    def _from_float(cls, value):
        """
        Convert an integer to a Float<nnn> instance.

        """
        sign = _math.copysign(1.0, value) < 0

        if _math.isnan(value):
            # XXX Think about transferring signaling bit and payload.
            return cls(
                type=_NAN,
                sign=sign,
            )

        if _math.isinf(value):
            return cls(type=_INFINITE, sign=sign)

        # Zeros
        if value == 0.0:
            return cls(
                type=_FINITE,
                sign=sign,
                exponent=cls._format.qmin,
                significand=0,
            )

        a, b = abs(value).as_integer_ratio()

        # compute exponent e for result; may be one too small in the case
        # that the rounded value of a/b lies in a different binade from a/b
        d = a.bit_length() - b.bit_length()
        d += (a >> d if d >= 0 else a << -d) >= b
        # Invariant: 2 ** (d - 1) <= a / b < 2 ** d.
        e = max(d - cls._format.precision, cls._format.qmin) - 2

        # approximate a/b by number of the form q * 2**e; adjust e if
        # necessary
        a, b = a << max(-e, 0), b << max(e, 0)
        q, r = divmod(a, b)
        q |= bool(r)

        return cls._format._final_round(sign, e, q)

    @classmethod
    def _from_int(cls, n):
        """
        Convert an integer to a Float<nnn> instance.

        """
        if n == 0:
            return cls(
                type=_FINITE,
                sign=False,
                exponent=cls._format.qmin,
                significand=0,
            )

        if n < 0:
            sign = 1
            n = -n
        else:
            sign = 0

        # Figure out exponent.
        e = max(n.bit_length() - cls._format.precision, cls._format.qmin) - 2

        shift = -e
        if shift >= 0:
            q = n << shift
        else:
            q = (n >> -shift) | bool(n & ~(-1 << -shift))

        return cls._format._final_round(sign, e, q)

    @classmethod
    def _from_str(cls, s):
        """
        Convert an input string to a Float<nnn> instance.

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
            a, b = intpart * 10 ** max(exp, 0), 10 ** max(0, -exp)

            # quick return for zeros
            if not a:
                return cls(
                    type=_FINITE,
                    sign=sign,
                    exponent=cls._format.qmin,
                    significand=0,
                )

            # compute exponent e for result; may be one too small in the case
            # that the rounded value of a/b lies in a different binade from a/b
            d = a.bit_length() - b.bit_length()
            d += (a >> d if d >= 0 else a << -d) >= b
            # Invariant: 2 ** (d - 1) <= a / b < 2 ** d.
            # The "- 2" gives us 2 extra bits to use for rounding.
            e = max(d - cls._format.precision, cls._format.qmin) - 2

            # approximate a/b by number of the form q * 2**e; adjust e if
            # necessary
            a, b = a << max(-e, 0), b << max(e, 0)
            q, r = divmod(a, b)
            # round to odd
            q |= bool(r)
            return cls._format._final_round(sign, e, q)

        elif m.group('infinite'):
            # Infinity.
            return cls(
                type=_INFINITE,
                sign=sign,
            )

        elif m.group('nan'):
            # NaN.
            signaling = bool(m.group('signaling'))

            # Parse payload, and clip to bounds if necessary.
            payload = int(m.group('payload') or 0)
            min_payload = 1 if signaling else 0
            max_payload = 2 ** (cls._format.precision - 2) - 1
            if payload < min_payload:
                payload = min_payload
            elif payload > max_payload:
                payload = max_payload

            return cls(
                type=_NAN,
                sign=sign,
                signaling=signaling,
                payload=payload,
            )
        else:
            assert False, "Shouldn't get here."

    @classmethod
    def decode(cls, encoded_value):
        """
        Decode a string of bytes to the corresponding Float<nnn> instance.

        """
        exponent_field_width = cls._format._exponent_field_width
        significand_field_width = cls._format.precision - 1

        # Extract fields.
        equivalent_int = _int_from_bytes(encoded_value)
        significand_field = equivalent_int & (2 ** significand_field_width - 1)
        equivalent_int >>= significand_field_width
        exponent_field = equivalent_int & (2 ** exponent_field_width - 1)
        equivalent_int >>= exponent_field_width
        sign = equivalent_int

        assert 0 <= exponent_field <= 2 ** exponent_field_width - 1
        assert 0 <= significand_field <= 2 ** significand_field_width - 1

        # Construct value.
        if exponent_field == 2 ** exponent_field_width - 1:
            # Infinities, Nans.
            if significand_field == 0:
                # Infinities.
                return cls(type=_INFINITE, sign=sign)
            else:
                # Nan.
                payload_width = significand_field_width - 1
                payload = significand_field & ((1 << payload_width) - 1)
                significand_field >>= payload_width
                # Top bit of significand field indicates whether this Nan is
                # quiet (1) or signaling (0).
                assert 0 <= significand_field <= 1
                signaling = not significand_field
                return cls(
                    type=_NAN,
                    sign=sign,
                    payload=payload,
                    signaling=signaling,
                )
        elif exponent_field == 0:
            # Subnormals, Zeros.
            return cls(
                type=_FINITE,
                sign=sign,
                exponent=cls._format.qmin,
                significand=significand_field,
            )
        else:
            significand = significand_field + 2 ** (cls._format.precision - 1)
            return cls(
                type=_FINITE,
                sign=sign,
                exponent=exponent_field - cls._format.qbias,
                significand=significand,
            )

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
                    self._significand - 2 ** (self._format.precision - 1)
                )
        elif self._type == _INFINITE:
            exponent_field = 2 ** exponent_field_width - 1
            significand_field = 0
        elif self._type == _NAN:
            exponent_field = 2 ** exponent_field_width - 1
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

    def negate(self):
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
            other = _Float64(other)
        elif isinstance(other, int):
            other = self._format(other)
        else:
            raise TypeError(
                "Can't convert operand {} of type {} to "
                "_BinaryFloatBase.".format(
                    other,
                    type(other),
                )
            )
        return other

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
