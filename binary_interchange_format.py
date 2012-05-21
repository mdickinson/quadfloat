import decimal as _decimal
import math as _math
import re as _re
import sys as _sys


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


class BinaryInterchangeFormat(object):
    _class__cache = {}

    def __init__(self, width):
        valid_width = width in {16, 32, 64} or width >= 128 and width % 32 == 0
        if not valid_width:
            raise ValueError(
                "For an interchange format, width should be 16, 32, 64, "
                "or a multiple of 32 >= 128."
            )
        self.width = width

    def __eq__(self, other):
        return self.width == other.width

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

    # XXX better name?
    @property
    def class_(self):
        if self not in BinaryInterchangeFormat._class__cache:
            class BinaryFormat(_BinaryFloatBase):
                _format = self
            BinaryFormat.__name__ = 'Float{}'.format(self.width)
            BinaryInterchangeFormat._class__cache[self] = BinaryFormat

        return BinaryInterchangeFormat._class__cache[self]

    def addition(self, source1, source2):
        return self.class_.addition(source1, source2)

    def multiplication(self, source1, source2):
        return self.class_.multiplication(source1, source2)

    def division(self, source1, source2):
        return self.class_.division(source1, source2)

    def decode(self, encoded_value):
        return self.class_.decode(encoded_value)


def _handle_overflow(cls, sign):
    """
    Handle an overflow.

    """
    # For now, just returns the appropriate infinity.  Someday this should
    # handle rounding modes, flags, etc.
    return cls(type=_INFINITE, sign=sign)


def _handle_invalid(cls, snan=None):
    """
    Handle an invalid operation.

    """
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
                raise ValueError("Unnormalized significand or exponent.")

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
            if not bool(signaling) <= payload < 2 ** (cls._format.precision - 2):
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
        XXX: could replace this with a comparison of corresponding byte strings.

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

        elif isinstance(value, int):
            # Initialize from an integer.
            return cls._from_int(value)

        elif isinstance(value, str):
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
            # XXX Think about transfering signaling bit and payload.
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
        e = max(d - cls._format.precision, cls._format.qmin)

        # approximate a/b by number of the form q * 2**e; adjust e if
        # necessary
        a, b = a << max(-e, 0), b << max(e, 0)
        q, r = divmod(a, b)

        assert q.bit_length() <= cls._format.precision
        if 2 * r > b or 2 * r == b and q & 1:
            q += 1
            if q.bit_length() == cls._format.precision + 1:
                q //= 2
                e += 1

        if e > cls._format.qmax:
            return _handle_overflow(cls, sign)

        return cls(
            type=_FINITE,
            sign=sign,
            exponent=e,
            significand=q,
        )

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
        e = n.bit_length() - cls._format.precision

        # q ~ n * 2**-e
        if e > 0:
            q = n >> e
            rtype = (
                2 * bool(n & (1 << (e - 1))) + bool(n & ((1 << (e - 1)) - 1))
            )
        else:
            q = n << -e
            rtype = 0

        assert q.bit_length() <= cls._format.precision

        # Round.
        if rtype == 3 or rtype == 2 and q & 1:
            q += 1
            if q.bit_length() == cls._format.precision + 1:
                q //= 2
                e += 1

        # Overflow.
        if e > cls._format.qmax:
            return _handle_overflow(cls, sign)

        return cls(
            type=_FINITE,
            sign=sign,
            exponent=e,
            significand=q,
        )

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
            e = max(d - cls._format.precision, cls._format.qmin)

            # approximate a/b by number of the form q * 2**e; adjust e if
            # necessary
            a, b = a << max(-e, 0), b << max(e, 0)
            q, r = divmod(a, b)

            assert q.bit_length() <= cls._format.precision
            if 2 * r > b or 2 * r == b and q & 1:
                q += 1
                if q.bit_length() == cls._format.precision + 1:
                    q //= 2
                    e += 1

            if e > cls._format.qmax:
                return _handle_overflow(cls, sign)

            return cls(
                type=_FINITE,
                sign=sign,
                exponent=e,
                significand=q,
            )

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
    def decode(cls, encoded_value, *, byteorder='little'):
        """
        Decode a string of bytes to the corresponding Float<nnn> instance.

        """
        exponent_field_width = cls._format._exponent_field_width
        significand_field_width = cls._format.precision - 1

        # Extract fields.
        equivalent_int = int.from_bytes(encoded_value, byteorder=byteorder)
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

    def encode(self, *, byteorder='little'):
        """
        Encode a Float<nnn> instance as a 16-character bytestring.

        """
        if self._type == _FINITE:

            # Exponent and significand fields.
            if self.is_subnormal() or self.is_zero():
                exponent_field = 0
                significand_field = self._significand
            else:
                exponent_field = self._exponent + self._format.qbias
                significand_field = (
                    self._significand - 2 ** (self._format.precision - 1)
                )

            exponent_field_width = self._format._exponent_field_width
            significand_field_width = self._format.precision - 1

            equivalent_int = (
                (self._sign << (exponent_field_width + significand_field_width)) +
                (exponent_field << (significand_field_width)) +
                significand_field
            )

            return equivalent_int.to_bytes(
                self._format.width // 8,
                byteorder=byteorder,
            )

        elif self._type == _INFINITE:

            exponent_field_width = self._format._exponent_field_width
            significand_field_width = self._format.precision - 1

            exponent_field = 2 ** exponent_field_width - 1
            significand_field = 0

            equivalent_int = (
                (self._sign << (exponent_field_width + significand_field_width)) +
                (exponent_field << (significand_field_width)) +
                significand_field
            )

            return equivalent_int.to_bytes(
                self._format.width // 8,
                byteorder=byteorder,
            )

        elif self._type == _NAN:

            exponent_field_width = self._format._exponent_field_width
            significand_field_width = self._format.precision - 1
            payload_width = significand_field_width - 1

            exponent_field = 2 ** exponent_field_width - 1
            significand_field = self._payload
            assert 0 <= self._payload < 2 ** payload_width

            significand_field = (
                ((not self._signaling) << payload_width) +
                self._payload
            )

            equivalent_int = (
                (self._sign << (exponent_field_width + significand_field_width)) +
                (exponent_field << (significand_field_width)) +
                significand_field
            )

            return equivalent_int.to_bytes(
                self._format.width // 8,
                byteorder=byteorder,
            )

        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

    # Binary operations.

    def subtraction(self, other):
        return self.addition(other.negate())

    @classmethod
    def _round_from_triple(cls, sign, exponent, significand):

        # Round the value significand * 2**exponent to the format.
        if significand == 0:
            return cls(
                type=_FINITE,
                sign=sign,
                exponent=cls._format.qmin,
                significand=0,
            )

        # ... first find exponent of result.

        # d satisfies 2**(d-1) <= significand * 2 ** exponent < 2**d.
        d = exponent + significand.bit_length()

        # Exponent of result.
        e = max(d - cls._format.precision, cls._format.qmin)

        # significand * 2**exponent ~ q * 2**e.
        # significand * 2**(exponent - e) ~ q
        shift = exponent - e
        if shift >= 0:
            q = significand << shift
            rtype = 0
        else:
            q = significand >> -shift
            r = significand & ((1 << -shift) - 1)

            # Classify r: 0 if exact, 2 if exact halfway, 1 / 3 for low / high
            if r > (1 << (-shift - 1)):
                rtype = 3
            elif r == (1 << (-shift - 1)):
                rtype = 2
            elif r > 0:
                rtype = 1
            else:
                rtype = 0

        assert q.bit_length() <= cls._format.precision

        # Round.
        if rtype == 3 or rtype == 2 and q & 1:
            q += 1
            if q.bit_length() == cls._format.precision + 1:
                q //= 2
                e += 1

        # Overflow.
        if e > cls._format.qmax:
            return _handle_overflow(cls, sign)

        return cls(
            type=_FINITE,
            sign=sign,
            exponent=e,
            significand=q,
        )

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

    @classmethod
    def addition(cls, self, other):
        if self._type == _FINITE:
            if other.is_infinite():
                return other

            if not other.is_finite():
                raise NotImplementedError(
                    "Addition for non-finite numbers not yet implemented."
                )

            # Finite case.
            #
            # XXX To do: optimize for the case that the exponents are
            # significantly different.  XXX. Sign is incorrect for some
            # rounding modes.

            exponent = min(self._exponent, other._exponent)
            significand = (
                (self._significand * (-1) ** self._sign <<
                 self._exponent - exponent) +
                (other._significand * (-1) ** other._sign <<
                 other._exponent - exponent)
            )
            sign = (significand < 0 or
                    significand == 0 and self._sign and other._sign)

            return cls._round_from_triple(
                sign=sign,
                exponent=exponent,
                significand=abs(significand),
            )

        elif self._type == _INFINITE:
            if other.is_nan():
                # infinity + nan -> nan
                if other.is_signaling():
                    return _handle_invalid(cls, snan=other)
                else:
                    return other

            elif other.is_infinite():
                # same sign -> infinity; opposite signs -> nan
                if self._sign == other._sign:
                    return self
                else:
                    return _handle_invalid(cls)

            else:
                return self

        elif self._type == _NAN:
            if self.is_signaling():
                return _handle_invalid(cls, snan=self)
            elif other.is_signaling():
                return _handle_invalid(cls, snan=other)
            else:
                return self

        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

    @classmethod
    def multiplication(cls, self, other):
        if self._type == _FINITE:

            if other.is_nan():
                if other.is_signaling():
                    return _handle_invalid(cls, snan=other)
                else:
                    # finite * nan -> nan
                    return cls(
                        type=_NAN,
                        sign=self._sign ^ other._sign,
                        payload=other._payload,
                    )

            elif other.is_infinite():
                if self.is_zero():
                    # zero * infinity -> nan
                    return _handle_invalid(cls)

                # non-zero finite * infinity -> infinity
                return cls(type=_INFINITE, sign=self._sign ^ other._sign)

            # finite * finite case.
            sign = self._sign ^ other._sign
            significand = self._significand * other._significand
            exponent = self._exponent + other._exponent

            return cls._round_from_triple(
                sign=sign,
                exponent=exponent,
                significand=significand,
            )

        elif self._type == _INFINITE:
            if other.is_nan():
                # infinity * nan -> nan
                if other.is_signaling():
                    return _handle_invalid(cls, snan=other)
                else:
                    return cls(
                        type=_NAN,
                        sign=self._sign ^ other._sign,
                        payload=other._payload,
                    )

            elif other.is_infinite() or not other.is_zero():
                # infinity * infinity -> infinity;
                # infinity * nonzero finite -> infinity
                return cls(type=_INFINITE, sign=self._sign ^ other._sign)

            elif other.is_zero():
                return _handle_invalid(cls)

        elif self._type == _NAN:
            if self.is_signaling():
                return _handle_invalid(cls, snan=self)
            elif other.is_signaling():
                return _handle_invalid(cls, snan=other)
            else:
                return cls(
                    type=_NAN,
                    sign=self._sign ^ other._sign,
                    payload=self._payload,
                )

        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

    @classmethod
    def division(cls, self, other):
        if self._type == _FINITE:

            if not other.is_finite():
                raise NotImplementedError("Division not yet implemented for non-finite numbers.")

            # Finite / finite case.
            sign = self._sign ^ other._sign

            if self.is_zero():
                if other.is_zero():
                    return _handle_invalid(cls)

                return cls(
                    type=_FINITE,
                    sign=sign,
                    exponent=self._format.qmin,
                    significand=0,
                )

            if other.is_zero():
                return cls(type=_INFINITE, sign=sign)

            # First find d such that 2**(d-1) <= abs(self) / abs(other) < 2**d.
            a = self._significand
            b = other._significand
            d = a.bit_length() - b.bit_length()
            d += (a >> d if d >= 0 else a << -d) >= b
            d += self._exponent - other._exponent

            # Exponent of result.
            e = max(d - self._format.precision, self._format.qmin)

            # Round (self / other) * 2**-e to nearest integer.
            # self / other * 2**-e == self._significand / other._significand * 2**shift, where...
            shift = self._exponent - other._exponent - e

            a, b = a << max(shift, 0), b << max(0, -shift)
            q, r = divmod(a, b)
            rtype = 2 * (2 * r > b) + (r != 0)

            # Now result approximated by (-1)**sign * q * 2**e.
            # Double check parameters.
            assert sign in (0, 1)
            assert e >= self._format.qmin
            assert q.bit_length() <= self._format.precision
            assert e == self._format.qmin or q.bit_length() == self._format.precision
            assert 0 <= rtype <= 3

            # Round.
            if rtype == 3 or rtype == 2 and q & 1:
                q += 1
                if q.bit_length() == self._format.precision + 1:
                    q //= 2
                    e += 1

            # Overflow.
            if e > self._format.qmax:
                return _handle_overflow(cls, sign)

            return cls(
                type=_FINITE,
                sign=sign,
                exponent=e,
                significand=q,
            )
        elif self._type == _INFINITE:

            raise NotImplementedError("Division not yet implemented for non-finite numbers.")


        elif self._type == _NAN:
            raise NotImplementedError("Division not yet implemented for non-finite numbers.")

        else:
            raise ValueError("invalid _type attribute: {}".format(self._type))

    def __neg__(self):
        return self.negate()

    def __abs__(self):
        return self.abs()
