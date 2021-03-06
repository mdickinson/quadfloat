from quadfloat.arithmetic import _divide_nearest
from quadfloat.api import (
    BinaryInterchangeFormat,
    binary16,
    binary32,
    binary64,
    BitString,
    encode,
    min_num,
    max_num,
    min_num_mag,
    max_num_mag,
)
from quadfloat.parsing import parse_finite_decimal
from quadfloat.tests.base_test_case import BaseTestCase


def string_to_decimal(s):
    # Normalized version of parse_finite_decimal.
    sign, exponent, coefficient = parse_finite_decimal(s)

    # Normalize: use an exponent of 0 for zeros, and
    # the largest exponent possible otherwise.
    if coefficient == 0:
        exponent = 0
    else:
        while coefficient % 10 == 0:
            coefficient //= 10
            exponent += 1
    return sign, exponent, coefficient


# binary16 details:
#
#    emax = 15
#    p = 11
#
#    11-bit precision
#     5-bit exponent, so normal range is 2 ** -14 through 2 ** 16.
#
#   next up from 1 is 1 + 2 ** -10
#   next down from 1 is 1 - 2 ** -11
#   max representable value is 2 ** 16 - 2 ** 5
#   smallest +ve integer that rounds to infinity is 2 ** 16 - 2 ** 4.
#   smallest +ve representable normal value is 2 ** -14.
#   smallest +ve representable value is 2 ** -24.
#   smallest +ve integer that can't be represented exactly is 2 ** 11 + 1


class TestBinary16(BaseTestCase):
    def test_construction_from_int(self):
        # Test round-half-to-even
        self.assertEqual(encode(binary16(2048)), BitString('0110100000000000'))
        self.assertEqual(encode(binary16(2049)), BitString('0110100000000000'))
        self.assertEqual(encode(binary16(2050)), BitString('0110100000000001'))
        self.assertEqual(encode(binary16(2051)), BitString('0110100000000010'))
        self.assertEqual(encode(binary16(2052)), BitString('0110100000000010'))
        self.assertEqual(encode(binary16(2053)), BitString('0110100000000010'))
        self.assertEqual(encode(binary16(2054)), BitString('0110100000000011'))
        self.assertEqual(encode(binary16(2055)), BitString('0110100000000100'))
        self.assertEqual(encode(binary16(2056)), BitString('0110100000000100'))

    def test_construction_from_float(self):
        self.assertInterchangeable(binary16(0.9), binary16('0.89990234375'))

        # Test round-half-to-even
        self.assertEqual(
            encode(binary16(2048.0)),
            BitString('0110100000000000'),
        )
        self.assertEqual(
            encode(binary16(2048.9999999999)),
            BitString('0110100000000000'),
        )
        self.assertEqual(
            encode(binary16(2049.0)),
            BitString('0110100000000000'),
        )
        self.assertEqual(
            encode(binary16(2049.0000000001)),
            BitString('0110100000000001'),
        )
        self.assertEqual(
            encode(binary16(2050.0)),
            BitString('0110100000000001'),
        )
        self.assertEqual(
            encode(binary16(2050.9999999999)),
            BitString('0110100000000001'),
        )
        self.assertEqual(
            encode(binary16(2051.0)),
            BitString('0110100000000010'),
        )
        self.assertEqual(
            encode(binary16(2051.0000000001)),
            BitString('0110100000000010'),
        )
        self.assertEqual(
            encode(binary16(2052.0)),
            BitString('0110100000000010'),
        )
        self.assertEqual(
            encode(binary16(2053.0)),
            BitString('0110100000000010'),
        )
        self.assertEqual(
            encode(binary16(2054.0)),
            BitString('0110100000000011'),
        )
        self.assertEqual(
            encode(binary16(2055.0)),
            BitString('0110100000000100'),
        )
        self.assertEqual(
            encode(binary16(2056.0)),
            BitString('0110100000000100'),
        )

        # Subnormals.
        eps = 1e-10
        tiny = 2.0 ** -24  # smallest positive representable binary16 subnormal
        test_values = [
            (0.0, BitString('0000000000000000')),
            (tiny * (0.5 - eps), BitString('0000000000000000')),
            (tiny * 0.5, BitString('0000000000000000')),  # halfway case
            (tiny * (0.5 + eps), BitString('0000000000000001')),
            (tiny, BitString('0000000000000001')),
            (tiny * (1.5 - eps), BitString('0000000000000001')),
            (tiny * 1.5, BitString('0000000000000010')),  # halfway case
            (tiny * (1.5 + eps), BitString('0000000000000010')),
            (tiny * 2.0, BitString('0000000000000010')),
            (tiny * (2.5 - eps), BitString('0000000000000010')),
            (tiny * 2.5, BitString('0000000000000010')),  # halfway case
            (tiny * (2.5 + eps), BitString('0000000000000011')),
            (tiny * 3.0, BitString('0000000000000011')),
        ]
        for x, bs in test_values:
            self.assertEqual(encode(binary16(x)), bs)

    def test_division(self):
        # Test division, with particular attention to correct rounding.
        # Integers up to 2048 all representable in binary16

        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4), binary32(4)),
            binary16(2048)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 1), binary32(4)),
            binary16(2048)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 2), binary32(4)),
            binary16(2048)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 3), binary32(4)),
            binary16(2048)
        )
        # Exact halfway case; should be rounded down.
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 4), binary32(4)),
            binary16(2048)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 5), binary32(4)),
            binary16(2050)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 6), binary32(4)),
            binary16(2050)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 7), binary32(4)),
            binary16(2050)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 8), binary32(4)),
            binary16(2050)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 9), binary32(4)),
            binary16(2050)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 10), binary32(4)),
            binary16(2050)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 11), binary32(4)),
            binary16(2050)
        )
        # Exact halfway case, rounds *up*!
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 12), binary32(4)),
            binary16(2052)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 13), binary32(4)),
            binary16(2052)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 14), binary32(4)),
            binary16(2052)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 15), binary32(4)),
            binary16(2052)
        )
        self.assertInterchangeable(
            binary16.division(binary32(2048 * 4 + 16), binary32(4)),
            binary16(2052)
        )

    def test_sqrt(self):
        # Easy small integer cases.
        for i in range(1, 46):
            self.assertInterchangeable(
                binary16.square_root(binary16(i * i)),
                binary16(i),
            )

        # Zeros.
        self.assertInterchangeable(
            binary16.square_root(binary16('0')),
            binary16('0'),
        )
        self.assertInterchangeable(
            binary16.square_root(binary16('-0')),
            binary16('-0'),
        )

        # Infinities
        self.assertInterchangeable(
            binary16.square_root(binary16('inf')),
            binary16('inf'),
        )

        self.assertInterchangeable(
            binary16.square_root(binary16('-inf')),
            binary16('nan'),
        )

        # Negatives
        self.assertInterchangeable(
            binary16.square_root(binary16('-4.0')),
            binary16('nan'),
        )

        # NaNs
        self.assertInterchangeable(
            binary16.square_root(binary16('snan(456)')),
            binary16('nan(456)'),
        )

        self.assertInterchangeable(
            binary16.square_root(binary16('-nan(123)')),
            binary16('-nan(123)'),
        )

        # Subnormal results.
        tiny = 2.0 ** -24  # smallest positive representable binary16 subnormal

        self.assertInterchangeable(
            binary16.square_root(binary64(tiny * tiny)),
            binary16(tiny),
        )
        self.assertInterchangeable(
            binary16.square_root(binary64(tiny * tiny * 0.25)),
            binary16('0.0'),
        )
        self.assertInterchangeable(
            binary16.square_root(binary64(tiny * tiny * 0.250000001)),
            binary16(tiny),
        )
        self.assertInterchangeable(
            binary16.square_root(binary64(tiny * tiny)),
            binary16(tiny),
        )
        self.assertInterchangeable(
            binary16.square_root(binary64(tiny * tiny * 2.24999999999)),
            binary16(tiny),
        )
        self.assertInterchangeable(
            binary16.square_root(binary64(tiny * tiny * 2.25)),
            binary16(2 * tiny),
        )
        self.assertInterchangeable(
            binary16.square_root(binary64(tiny * tiny * 2.250000001)),
            binary16(2 * tiny),
        )
        self.assertInterchangeable(
            binary16.square_root(binary64(tiny * tiny * 4.0)),
            binary16(2 * tiny),
        )

    def test_repr_construct_roundtrip(self):
        # Roundtrip tests.

        # Particularly interesting values.
        test_values = [
            # sits at the middle of a *closed* interval with one endpoint at
            # 4150; so '415e1' is an acceptable short representation.
            binary16('4152'),
            # sits at the middle of an *open* interval with one endpoint at
            # 4150; so '415e1' is *not* an acceptable short representation.
            binary16('4148'),
            # power of 2;  interval needs special casing.
            binary16('0.0078125'),
            # another power of 2 where rounding to nearest for the best
            # final digit produces a value out of range.
            binary16('0.015625')
        ]

        # With binary16, it's feasible to test *all* the values.
        for value_as_int in range(2**16):
            value = binary16.decode(
                BitString.from_int(width=16, value_as_int=value_as_int))
            test_values.append(value)

        for value in test_values:
            repr_value = repr(value)
            reconstructed_value = eval(repr_value)
            self.assertInterchangeable(value, reconstructed_value)

            str_value = str(value)
            reconstructed_value = binary16(str_value)
            self.assertInterchangeable(value, reconstructed_value)

    def test_short_float_repr(self):
        # Test that we're giving back the *shortest* representation.

        # First, just tests for the value represented by the output string,
        # rather than the string representation itself (exact positioning of
        # decimal point, exponent, etc.)

        TINY = 2.0 ** -24
        test_pairs = [
            (binary16(TINY), '6e-8'),
            (binary16(2 * TINY), '1e-7'),
            (binary16(3 * TINY), '2e-7'),
            (binary16(4 * TINY), '2.4e-7'),
            (binary16(5 * TINY), '3e-7'),
            (binary16('0.015625'), '0.01563'),
            (binary16('1.23'), '1.23'),
            (binary16('4152'), '4.15e+3'),
            (binary16('4148'), '4.148e+3'),
            (binary16('4112'), '4.11e+3'),
        ]
        for input, output_string in test_pairs:
            input_string = str(input)
            self.assertEqual(
                string_to_decimal(input_string),
                string_to_decimal(output_string),
            )

        # Exhaustive testing for 3-digit decimal -> binary16 -> decimal
        # round-tripping.

        # The mapping from 3-digit decimal strings to binary16 objects
        # is injective, outside of the overflow / underflow regions.
        # (Key point in the proof is that 2 ** 10 < 10 ** 3).  So 3-digit
        # strings should roundtrip.

        # Subnormals: tiny value for binary16 is 2 ** -24, or around
        # 5.9e-08.  So increments of 1e-07 should be safe.
        def input_strings():
            for exp in range(-7, 2):
                for n in range(1000):
                    yield '{0}e{1}'.format(n, exp)
                    yield '-{0}e{1}'.format(n, exp)
            for exp in range(2, 3):
                for n in range(656):
                    yield '{0}e{1}'.format(n, exp)
                    yield '-{0}e{1}'.format(n, exp)

        for input_string in input_strings():
            output_string = str(binary16(input_string))
            self.assertEqual(
                input_string.startswith('-'),
                output_string.startswith('-'),
            )
            self.assertEqual(
                string_to_decimal(input_string),
                string_to_decimal(output_string),
            )

    def test_short_float_repr_subnormal_corner_case(self):
        # Corner case: the naive algorithm establishes the exponent first and
        # then outputs the shortest string for that exponent; this can fail to
        # produce the closest shortest string in the case where the closest
        # shortest string has a different exponent.  Some thought shows that
        # this is only a problem when the interval of values rounding to the
        # particular target value contains a subinterval of the form [9, 10] *
        # 10 ** e for some e.  For the standard formats, an interval of this
        # relative width can only occur for subnormal target values.

        # This corner case doesn't occur for binary16, binary32, binary64 or
        # binary128.

        width = 16
        format = BinaryInterchangeFormat(width)
        TINY = format.decode(BitString('0000000000000001'))
        for n in range(1, 100):
            binary_value = n * TINY

            # Find the closest 1-digit string to binary_value.
            # XXX Use string formatting for this, once it's implemented.
            a = binary_value._significand << max(binary_value._exponent, 0)
            b = 1 << max(0, -binary_value._exponent)
            n = len(str(a)) - len(str(b))
            n += (a // 10 ** n if n >= 0 else a * 10 ** -n) >= b

            # Invariant: 10 ** (n-1) <= abs(binary_value) <= 10 ** n.
            # Want a single place, so binary_value * 10 ** 1-n
            a *= 10 ** max(1 - n, 0)
            b *= 10 ** max(0, n - 1)
            assert b <= a < 10 * b
            best_digit = _divide_nearest(a, b)
            assert 1 <= best_digit <= 10
            best_str = '{0}e{1}'.format(best_digit, n - 1)

            # If the returned string has only one digit, it should be
            # equal to the closest string.
            output_str = str(binary_value)
            _, _, output_coeff = string_to_decimal(output_str)
            if len(str(output_coeff)) == 1:
                self.assertEqual(
                    string_to_decimal(output_str),
                    string_to_decimal(best_str),
                )

    def _comparison_test_values(self):
        zeros = [binary16('0.0'), binary16('-0.0'), 0]
        # List of lists;  all values in each of the inner lists are
        # equal; outer list ordered numerically.
        positives = [
            [0.4999999999],
            [binary16(0.5), 0.5],
            [binary16(1 - 2 ** -11), 1.0 - 2.0 ** -11],
            [1.0 - 2.0 ** -12],
            [binary64(1 - 2.0 ** -53), 1 - 2.0 ** -53],
            [binary16('1.0'), 1, 1.0],
            [binary64(1 + 2.0 ** -52), 1 + 2.0 ** -52],
            [1.0 + 2.0 ** -11],
            [binary16(1 + 2 ** -10), 1 + 2.0 ** -10],
            [binary16(1.5)],
            [binary16('2.0'), 2],
            [binary16(2 ** 11 - 1), 2 ** 11 - 1],
            [binary16(2 ** 11), 2 ** 11],
            [2 ** 11 + 1],
            [binary16(2 ** 16 - 2 ** 5), 2 ** 16 - 2 ** 5],
            [2 ** 16 - 2 ** 4 - 1],
            [2 ** 16 - 2 ** 4],
            [2 ** 16],
            [binary16('inf')],
        ]
        negatives = [
            [-x for x in sublist]
            for sublist in positives
        ]

        all_pairs = list(reversed(negatives)) + [zeros] + positives

        for i, xset in enumerate(all_pairs):
            for yset in all_pairs[:i]:
                for x in xset:
                    for y in yset:
                        yield x, y, 'GT'
                        yield y, x, 'LT'
            yset = all_pairs[i]
            for x in xset:
                for y in yset:
                    yield x, y, 'EQ'

        # quiet nans
        nans = [binary16('nan'), binary16('-nan(123)')]
        for xset in all_pairs:
            for x in xset:
                for y in nans:
                    yield x, y, 'UN'
                    yield y, x, 'UN'
        for x in nans:
            for y in nans:
                yield x, y, 'UN'

        # signaling nans
        snans = [binary16('-snan'), binary16('snan(456)')]
        for xset in all_pairs + [nans]:
            for x in xset:
                for y in snans:
                    yield x, y, 'SI'
                    yield y, x, 'SI'
        for x in snans:
            for y in snans:
                yield x, y, 'SI'

    def test_rich_comparison_operators(self):
        # Test overloads for __eq__, __lt__, etc.
        msgfmt = '{0!r} {1!r} {2!r}'
        for x, y, reln in self._comparison_test_values():
            if reln == 'EQ':
                self.assertTrue(x == y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x != y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x < y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x > y, msg=msgfmt.format(x, y, reln))
                self.assertTrue(x <= y, msg=msgfmt.format(x, y, reln))
                self.assertTrue(x >= y, msg=msgfmt.format(x, y, reln))
                self.assertEqual(hash(x), hash(y))
            elif reln == 'LT':
                self.assertFalse(x == y, msg=msgfmt.format(x, y, reln))
                self.assertTrue(x != y, msg=msgfmt.format(x, y, reln))
                self.assertTrue(x < y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x > y, msg=msgfmt.format(x, y, reln))
                self.assertTrue(x <= y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x >= y, msg=msgfmt.format(x, y, reln))
            elif reln == 'GT':
                self.assertFalse(x == y, msg=msgfmt.format(x, y, reln))
                self.assertTrue(x != y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x < y, msg=msgfmt.format(x, y, reln))
                self.assertTrue(x > y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x <= y, msg=msgfmt.format(x, y, reln))
                self.assertTrue(x >= y, msg=msgfmt.format(x, y, reln))
            elif reln == 'UN':
                self.assertFalse(x == y, msg=msgfmt.format(x, y, reln))
                self.assertTrue(x != y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x < y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x > y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x <= y, msg=msgfmt.format(x, y, reln))
                self.assertFalse(x >= y, msg=msgfmt.format(x, y, reln))
            elif reln == 'SI':
                with self.assertRaises(ValueError):
                    x == y
                with self.assertRaises(ValueError):
                    x != y
                with self.assertRaises(ValueError):
                    x < y
                with self.assertRaises(ValueError):
                    x > y
                with self.assertRaises(ValueError):
                    x <= y
                with self.assertRaises(ValueError):
                    x >= y

    def test_hash(self):
        test_strings = ['inf', '-inf', 'nan', '-nan', '0.0', '-0.0',
                        '1.0', '0.125', '-1.0', '-2.0', '-1024.0']

        for test_string in test_strings:
            self.assertEqual(
                hash(binary16(test_string)),
                hash(float(test_string)),
            )

        # Signaling NaNs can't be hashed.
        snan = binary16('snan')
        with self.assertRaises(ValueError):
            hash(snan)

    def test_min_num(self):
        x = binary16('2.3')
        y = binary32('1.0')
        with self.assertRaises(ValueError):
            min_num(x, y)

    def test_max_num(self):
        x = binary16('2.3')
        y = binary32('1.0')
        with self.assertRaises(ValueError):
            max_num(x, y)

    def test_min_num_mag(self):
        x = binary16('2.3')
        y = binary32('1.0')
        with self.assertRaises(ValueError):
            min_num_mag(x, y)

    def test_max_num_mag(self):
        x = binary16('2.3')
        y = binary32('1.0')
        with self.assertRaises(ValueError):
            max_num_mag(x, y)
