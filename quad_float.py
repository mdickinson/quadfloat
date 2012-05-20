import decimal
import math
import re
import sys


_BINARY_INTERCHANGE_FORMAT_PRECISIONS = {
    16: 11,
    32: 24,
}


_number_parser = re.compile(r"""        # A numeric string consists of:
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
""", re.VERBOSE | re.IGNORECASE).match


class BinaryInterchangeFormat(object):
    def __init__(self, width):
        valid_width = width in {16, 32, 64} or width >= 128 and width % 32 == 0
        if not valid_width:
            raise ValueError(
                "For an interchange format, width should be 16, 32, 64, "
                "or a multiple of 32 >= 128."
            )
        self.width = width

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


binary16 = BinaryInterchangeFormat(width=16)
binary32 = BinaryInterchangeFormat(width=32)
binary64 = BinaryInterchangeFormat(width=64)
binary128 = BinaryInterchangeFormat(width=128)


def _handle_overflow(sign):
    """
    Handle an overflow.

    """
    # For now, just returns the appropriate infinity.  Someday this should
    # handle rounding modes, flags, etc.
    return InfiniteQuadFloat(sign=sign)


def _handle_invalid(snan=None):
    """
    Handle an invalid operation.

    """
    if snan is not None:
        # Invalid operation from an snan: quiet the sNaN.
        return NanQuadFloat(
            sign=snan._sign,
            payload=snan._payload,
            signaling=False,
        )

    # For now, just return a quiet NaN.  Someday this should be more
    # sophisticated.
    return NanQuadFloat()


class QuadFloat(object):
    _format = binary128

    def __new__(cls, value=0):
        """
        QuadFloat([value])

        Create a new QuadFloat instance from the given input.

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
                "Cannot construct a QuadFloat instance from a "
                "value of type {}".format(type(value))
            )

    def __repr__(self):
        return "QuadFloat('{}')".format(self._to_str())

    def __str__(self):
        return self._to_str()

    def is_sign_minus(self):
        """
        Return True if self has a negative sign.  This applies to zeros and
        NaNs as well as infinities and finite numbers.

        """
        return self._sign

    @classmethod
    def _from_float(cls, value):
        """
        Convert an integer to a QuadFloat instance.

        """
        sign = math.copysign(1.0, value) < 0

        if math.isnan(value):
            # XXX Think about transfering signaling bit and payload.
            return NanQuadFloat(sign=sign)

        if math.isinf(value):
            return InfiniteQuadFloat(sign=sign)

        # Zeros
        if value == 0.0:
            return FiniteQuadFloat(
                sign=sign,
                exponent=cls._format.qmin,
                significand=0,
            )

        # Now value is finite;  abs(value) = m * 2**e.
        m, e = math.frexp(math.fabs(value))
        m *= 2 ** sys.float_info.mant_dig
        e -= sys.float_info.mant_dig
        assert m == int(m)
        m = int(m)

        # Normalize.  Note that all floats (including subnormals) will be
        # normal QuadFloats.
        shift = cls._format.precision - m.bit_length()
        m, e = m << shift, e - shift

        return FiniteQuadFloat(
            sign=sign,
            exponent=e,
            significand=m,
        )

    @classmethod
    def _from_int(cls, n):
        """
        Convert an integer to a QuadFloat instance.

        """
        if n == 0:
            return FiniteQuadFloat(
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
            rtype = 2 * bool(n & (1 << (e-1))) + bool(n & ((1 << (e-1)) - 1))
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
            return _handle_overflow(sign)

        return FiniteQuadFloat(
            sign=sign,
            exponent=e,
            significand=q,
        )

    @classmethod
    def _from_str(cls, s):
        """
        Convert an input string to a QuadFloat instance.

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
                return FiniteQuadFloat(
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
                return _handle_overflow(sign)

            return FiniteQuadFloat(
                sign=sign,
                exponent=e,
                significand=q,
            )

        elif m.group('infinite'):
            # Infinity.
            return InfiniteQuadFloat(
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

            return NanQuadFloat(
                sign=sign,
                signaling=signaling,
                payload=payload,
            )
        else:
            assert False, "Shouldn't get here."

    @classmethod
    def decode(cls, encoded_value, *, byteorder='little'):
        """
        Decode a string of bytes to the corresponding QuadFloat instance.

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
                return InfiniteQuadFloat(sign=sign)
            else:
                # Nan.
                payload_width = significand_field_width - 1
                payload = significand_field & ((1 << payload_width) - 1)
                significand_field >>= payload_width
                # Top bit of significand field indicates whether this Nan is
                # quiet (1) or signaling (0).
                assert 0 <= significand_field <= 1
                signaling = not significand_field
                return NanQuadFloat(
                    sign=sign,
                    payload=payload,
                    signaling=signaling,
                )
        elif exponent_field == 0:
            # Subnormals, Zeros.
            return FiniteQuadFloat(
                sign=sign,
                exponent=cls._format.qmin,
                significand=significand_field,
            )
        else:
            significand = significand_field + 2 ** (cls._format.precision - 1)
            return FiniteQuadFloat(
                sign=sign,
                exponent=exponent_field - cls._format.qbias,
                significand=significand,
            )

    def encode(self, *, byteorder='little'):
        """
        Encode a QuadFloat instance as a 16-character bytestring.

        """
        raise NotImplementedError(
            "QuadFloat.encode is implemented by concrete subclasses."
        )


class FiniteQuadFloat(QuadFloat):
    def __new__(cls, sign, exponent, significand):
        """
        Create a finite QuadFloat from an integer triple.

        Returns the QuadFloat with value

            (-1) ** sign * 2 ** exponent * significand

        """
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
        self._sign = bool(sign)
        self._exponent = int(exponent)
        self._significand = int(significand)

        return self

    def _equivalent(self, other):
        return (
            isinstance(other, FiniteQuadFloat) and
            self._sign == other._sign and
            self._exponent == other._exponent and
            self._significand == other._significand
        )

    def _to_str(self, places=None):
        """
        Convert to a Decimal string with a given
        number of significant digits.

        """
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
            decimal.Decimal(
                '{0}{1}e{2}'.format(
                    '-' if self._sign else '',
                    q,
                    m,
                )
            )
        )

    def is_finite(self):
        return True

    def is_infinite(self):
        return False

    def is_nan(self):
        return False

    def is_signaling(self):
        return False

    def is_zero(self):
        return self._significand == 0

    def _is_subnormal_or_zero(self):
        """
        Return True if this instance is subnormal or zero, else False.

        """
        return self._significand < 2 ** (self._format.precision - 1)

    def encode(self, *, byteorder='little'):
        """
        Encode a FiniteQuadFloat instance as a 16-character bytestring.

        """
        # Exponent and significand fields.
        if self._is_subnormal_or_zero():
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

    # Arithmetic operations.

    def __add__(self, other):
        if other.is_infinite():
            return other

        if not other.is_finite():
            raise NotImplementedError(
                "Addition for non-finite numbers not yet implemented."
            )

        # Finite case.
        #
        # XXX To do: optimize for the case that the exponents are significantly
        # different.
        # XXX. Sign is incorrect for some rounding modes.

        exponent = min(self._exponent, other._exponent)
        significand = (
            (self._significand * (-1) ** self._sign <<
             self._exponent - exponent) +
            (other._significand * (-1) ** other._sign <<
             other._exponent - exponent)
        )
        sign = (significand < 0 or
                significand == 0 and self._sign and other._sign)

        return FiniteQuadFloat._round_from_triple(
            sign=sign,
            exponent=exponent,
            significand=abs(significand),
        )

    def __mul__(self, other):
        if other.is_nan():

            if other.is_signaling():
                return _handle_invalid(snan = other)
            else:
                # finite * nan -> nan
                return NanQuadFloat(
                    sign=self._sign ^ other._sign,
                    payload=other._payload,
                )

        elif other.is_infinite():
            if self.is_zero():
                # zero * infinity -> nan
                return _handle_invalid()

            # non-zero finite * infinity -> infinity
            return InfiniteQuadFloat(sign=self._sign ^ other._sign)

        # finite * finite case.
        sign = self._sign ^ other._sign
        significand = self._significand * other._significand
        exponent = self._exponent + other._exponent

        return FiniteQuadFloat._round_from_triple(
            sign=sign,
            exponent=exponent,
            significand=significand,
        )

    @classmethod
    def _round_from_triple(self, sign, exponent, significand):

        # Round the value significand * 2**exponent to the format.
        if significand == 0:
            return FiniteQuadFloat(
                sign=sign,
                exponent=self._format.qmin,
                significand=0,
            )

        # ... first find exponent of result.

        # d satisfies 2**(d-1) <= significand * 2 ** exponent < 2**d.
        d = exponent + significand.bit_length()

        # Exponent of result.
        e = max(d - self._format.precision, self._format.qmin)

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

        assert q.bit_length() <= self._format.precision

        # Round.
        if rtype == 3 or rtype == 2 and q & 1:
            q += 1
            if q.bit_length() == self._format.precision + 1:
                q //= 2
                e += 1

        # Overflow.
        if e > self._format.qmax:
            return _handle_overflow(sign)

        return FiniteQuadFloat(
            sign=sign,
            exponent=e,
            significand=q,
        )


class InfiniteQuadFloat(QuadFloat):
    def __new__(cls, sign):
        self = object.__new__(cls)
        self._sign = bool(sign)
        return self

    def _equivalent(self, other):
        return (
            isinstance(other, InfiniteQuadFloat) and
            self._sign == other._sign
        )

    def _to_str(self):
        return '-Infinity' if self._sign else 'Infinity'

    def is_finite(self):
        return False

    def is_infinite(self):
        return True

    def is_nan(self):
        return False

    def is_signaling(self):
        return False

    def is_zero(self):
        return False

    # Arithmetic operations.

    def __add__(self, other):
        if other.is_nan():
            # infinity + nan -> nan
            if other.is_signaling():
                return _handle_invalid(snan=other)
            else:
                return other

        elif other.is_infinite():
            # same sign -> infinity; opposite signs -> nan
            if self._sign == other._sign:
                return self
            else:
                return _handle_invalid()

        else:
            return self

    def __mul__(self, other):
        if other.is_nan():
            # infinity * nan -> nan
            if other.is_signaling():
                return _handle_invalid(snan=other)
            else:
                return NanQuadFloat(
                    sign=self._sign ^ other._sign,
                    payload=other._payload,
                )

        elif other.is_infinite() or not other.is_zero():
            # infinity * infinity -> infinity;
            # infinity * nonzero finite -> infinity
            return InfiniteQuadFloat(sign=self._sign ^ other._sign)

        elif other.is_zero():
            return _handle_invalid()

    def encode(self, *, byteorder='little'):
        """
        Encode an InfiniteQuadFloat instance as a 16-character bytestring.

        """
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


class NanQuadFloat(QuadFloat):
    def __new__(cls, sign=False, payload=0, signaling=False):
        # Payload must be at least 1 for a signaling nan.
        if not bool(signaling) <= payload < 2 ** (cls._format.precision - 2):
            raise ValueError("NaN payload out of range.")

        self = object.__new__(cls)
        self._sign = bool(sign)
        self._payload = int(payload)
        self._signaling = bool(signaling)
        return self

    # Arithmetic operations.

    def __add__(self, other):
        if self.is_signaling():
            return _handle_invalid(snan=self)
        elif other.is_signaling():
            return _handle_invalid(snan=other)
        else:
            return self

    def __mul__(self, other):
        if self.is_signaling():
            return _handle_invalid(snan=self)
        elif other.is_signaling():
            return _handle_invalid(snan=other)
        else:
            return NanQuadFloat(
                sign=self._sign ^ other._sign,
                payload=self._payload,
            )

    def _equivalent(self, other):
        return (
            isinstance(other, NanQuadFloat) and
            self._sign == other._sign and
            self._payload == other._payload and
            self._signaling == other._signaling
        )

    def _to_str(self):
        pieces = []
        if self._sign:
            pieces.append('-')
        if self._signaling:
            pieces.append('s')
        pieces.append('NaN')
        pieces.append('({})'.format(self._payload))
        return ''.join(pieces)

    def is_finite(self):
        return False

    def is_infinite(self):
        return False

    def is_nan(self):
        return True

    def is_signaling(self):
        return self._signaling

    def is_zero(self):
        return False

    def encode(self, *, byteorder='little'):
        """
        Encode a NanQuadFloat instance as a 16-character bytestring.

        """
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
