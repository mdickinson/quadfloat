import decimal
import unittest

from binary_interchange_format import BinaryInterchangeFormat
from binary_interchange_format import _bytes_from_iterable, _divide_nearest


Float16 = BinaryInterchangeFormat(width=16)
Float32 = BinaryInterchangeFormat(width=32)
Float64 = BinaryInterchangeFormat(width=64)


# Float16 details:
#
#    11-bit precision
#     5-bit exponent, so normal range is 2**-14 through 2**16.
#
#   next up from 1 is 1 + 2**-10
#   next down from 1 is 1 - 2**-11
#   max representable value is 2**16 - 2**5
#   smallest +ve integer that rounds to infinity is 2**16 - 2**4.
#   smallest +ve representable normal value is 2**-14.
#   smallest +ve representable value is 2**-24.
#   smallest +ve integer that can't be represented exactly is 2**11 + 1


class TestFloat16(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two Float16 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertTrue(quad1._equivalent(quad2),
                        msg = '{!r} not equivalent to {!r}'.format(quad1, quad2))

    def test_construction_from_int(self):
        # Test round-half-to-even

        # 2048 -> significand bits of 0, exponent of ???
        # 5 exponent bits;  for 1.0, would expect exponent bits to have value 15
        # so for 2048.0, should be 15+ 11 = 26.  Shift by 2 to get 104.
        self.assertEqual(Float16(2048).encode(), b'\x00\x68')
        self.assertEqual(Float16(2049).encode(), b'\x00\x68')
        self.assertEqual(Float16(2050).encode(), b'\x01\x68')
        self.assertEqual(Float16(2051).encode(), b'\x02\x68')
        self.assertEqual(Float16(2052).encode(), b'\x02\x68')
        self.assertEqual(Float16(2053).encode(), b'\x02\x68')
        self.assertEqual(Float16(2054).encode(), b'\x03\x68')
        self.assertEqual(Float16(2055).encode(), b'\x04\x68')
        self.assertEqual(Float16(2056).encode(), b'\x04\x68')

    def test_construction_from_float(self):
        self.assertInterchangeable(Float16(0.9), Float16('0.89990234375'))

        # Test round-half-to-even

        self.assertEqual(Float16(2048.0).encode(), b'\x00\x68')
        # halfway case
        self.assertEqual(Float16(2048.9999999999).encode(), b'\x00\x68')
        self.assertEqual(Float16(2049.0).encode(), b'\x00\x68')
        self.assertEqual(Float16(2049.0000000001).encode(), b'\x01\x68')
        self.assertEqual(Float16(2050.0).encode(), b'\x01\x68')
        self.assertEqual(Float16(2050.9999999999).encode(), b'\x01\x68')
        self.assertEqual(Float16(2051.0).encode(), b'\x02\x68')
        self.assertEqual(Float16(2051.0000000001).encode(), b'\x02\x68')
        self.assertEqual(Float16(2052.0).encode(), b'\x02\x68')
        self.assertEqual(Float16(2053.0).encode(), b'\x02\x68')
        self.assertEqual(Float16(2054.0).encode(), b'\x03\x68')
        self.assertEqual(Float16(2055.0).encode(), b'\x04\x68')
        self.assertEqual(Float16(2056.0).encode(), b'\x04\x68')

        # Subnormals.
        eps = 1e-10
        tiny = 2.0**-24  # smallest positive representable float16 subnormal
        test_values = [
            (0.0, b'\x00\x00'),
            (tiny * (0.5 - eps), b'\x00\x00'),
            (tiny * 0.5, b'\x00\x00'),  # halfway case
            (tiny * (0.5 + eps), b'\x01\x00'),
            (tiny, b'\x01\x00'),
            (tiny * (1.5 - eps), b'\x01\x00'),
            (tiny * 1.5, b'\x02\x00'),  # halfway case
            (tiny * (1.5 + eps), b'\x02\x00'),
            (tiny * 2.0, b'\x02\x00'),
            (tiny * (2.5 - eps), b'\x02\x00'),
            (tiny * 2.5, b'\x02\x00'),  # halfway case
            (tiny * (2.5 + eps), b'\x03\x00'),
            (tiny * 3.0, b'\x03\x00'),
        ]
        for x, bs in test_values:
            self.assertEqual(Float16(x).encode(), bs)

    def test_division(self):
        # Test division, with particular attention to correct rounding.
        # Integers up to 2048 all representable in Float16

        self.assertInterchangeable(
            Float16.division(Float32(2048*4), Float32(4)),
            Float16(2048)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 1), Float32(4)),
            Float16(2048)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 2), Float32(4)),
            Float16(2048)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 3), Float32(4)),
            Float16(2048)
        )
        # Exact halfway case; should be rounded down.
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 4), Float32(4)),
            Float16(2048)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 5), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 6), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 7), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 8), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 9), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 10), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 11), Float32(4)),
            Float16(2050)
        )
        # Exact halfway case, rounds *up*!
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 12), Float32(4)),
            Float16(2052)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 13), Float32(4)),
            Float16(2052)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 14), Float32(4)),
            Float16(2052)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 15), Float32(4)),
            Float16(2052)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 16), Float32(4)),
            Float16(2052)
        )

    def test_sqrt(self):
        # Easy small integer cases.
        for i in range(1, 46):
            self.assertInterchangeable(
                Float16.square_root(Float16(i * i)),
                Float16(i),
            )

        # Zeros.
        self.assertInterchangeable(
            Float16.square_root(Float16('0')),
            Float16('0'),
        )
        self.assertInterchangeable(
            Float16.square_root(Float16('-0')),
            Float16('-0'),
        )

        # Infinities
        self.assertInterchangeable(
            Float16.square_root(Float16('inf')),
            Float16('inf'),
        )

        self.assertInterchangeable(
            Float16.square_root(Float16('-inf')),
            Float16('nan'),
        )

        # Negatives
        self.assertInterchangeable(
            Float16.square_root(Float16('-4.0')),
            Float16('nan'),
        )

        # NaNs
        self.assertInterchangeable(
            Float16.square_root(Float16('snan(456)')),
            Float16('nan(456)'),
        )

        self.assertInterchangeable(
            Float16.square_root(Float16('-nan(123)')),
            Float16('-nan(123)'),
        )

        # Subnormal results.
        tiny = 2.0**-24  # smallest positive representable float16 subnormal

        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny)),
            Float16(tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 0.25)),
            Float16('0.0'),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 0.250000001)),
            Float16(tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny)),
            Float16(tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 2.24999999999)),
            Float16(tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 2.25)),
            Float16(2 * tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 2.250000001)),
            Float16(2 * tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 4.0)),
            Float16(2 * tiny),
        )

    def test_repr_construct_roundtrip(self):
        # Roundtrip tests.

        # Particularly interesting values.
        test_values = [
            Float16('4152'),  # sits at the middle of a *closed* interval with one endpoint at 4150;
                              # so '415e1' is an acceptable short representation.
            Float16('4148'),  # sits at the middle of an *open* interval with one endpoint at 4150;
                              # so '415e1' is *not* an acceptable short representation.
            Float16('0.0078125'),  # power of 2;  interval needs special casing.
            Float16('0.015625')  # another power of 2 where rounding to nearest for the best
                                # final digit produces a value out of range.
        ]

        # With Float16, it's feasible to test *all* the values.
        for high_byte in range(256):
            for low_byte in range(256):
                value = Float16.decode(_bytes_from_iterable([low_byte, high_byte]))
                test_values.append(value)

        for value in test_values:
            repr_value = repr(value)
            reconstructed_value = eval(repr_value)
            self.assertInterchangeable(value, reconstructed_value)

            str_value = str(value)
            reconstructed_value = Float16(str_value)
            self.assertInterchangeable(value, reconstructed_value)

    def test_short_float_repr(self):
        # Test that we're giving back the *shortest* representation.

        # First, just tests for the value represented by the output string,
        # rather than the string representation itself (exact positioning of
        # decimal point, exponent, etc.)

        TINY = 2.0**-24
        test_pairs = [
            (Float16(TINY), '6e-8'),
            (Float16(2 * TINY), '1e-7'),
            (Float16(3 * TINY), '2e-7'),
            (Float16(4 * TINY), '2.4e-7'),
            (Float16(5 * TINY), '3e-7'),
            (Float16('0.015625'), '0.01563'),
            (Float16('1.23'), '1.23'),
            (Float16('4152'), '415e1'),
            (Float16('4148'), '4148'),
        ]
        for input, output_string in test_pairs:
            input_string = str(input)
            self.assertEqual(
                decimal.Decimal(input_string),
                decimal.Decimal(output_string),
            )

        # Exhaustive testing for 3-digit decimal -> float16 -> decimal
        # round-tripping.

        # The mapping from 3-digit decimal strings to float16 objects
        # is injective, outside of the overflow / underflow regions.
        # (Key point in the proof is that 2**10 < 10**3).  So 3-digit
        # strings should roundtrip.

        # Subnormals: tiny value for float16 is 2**-24, or around
        # 5.9e-08.  So increments of 1e-07 should be safe.
        def input_strings():
            for exp in range(-7, 2):
                for n in range(1000):
                    yield '{}e{}'.format(n, exp)
                    yield '-{}e{}'.format(n, exp)
            for exp in range(2, 3):
                for n in range(656):
                    yield '{}e{}'.format(n, exp)
                    yield '-{}e{}'.format(n, exp)

        for input_string in input_strings():
            output_string = str(Float16(input_string))
            self.assertEqual(
                input_string.startswith('-'),
                output_string.startswith('-'),
            )
            self.assertEqual(
                decimal.Decimal(input_string),
                decimal.Decimal(output_string),
            )

    def test_short_float_repr_subnormal_corner_case(self):
        # Corner case: the naive algorithm establishes the exponent first and
        # then outputs the shortest string for that exponent; this can fail to
        # produce the closest shortest string in the case where the closest
        # shortest string has a different exponent.  Some thought shows that
        # this is only a problem when the interval of values rounding to the
        # particular target value contains a subinterval of the form [9, 10] *
        # 10**e for some e.  For the standard formats, an interval of this
        # relative width can only occur for subnormal target values.

        # This corner case doesn't occur for float16, float32, float64 or
        # float128.

        width = 16
        format = BinaryInterchangeFormat(width)
        TINY = format.decode(b'\x01' + b'\x00' * (width // 8 - 1))
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
            best_str = '{}e{}'.format(best_digit, n - 1)

            # If the returned string has only one digit, it should be
            # equal to the closest string.
            output_str = str(binary_value)
            if len(decimal.Decimal(output_str)._int.strip('0')) == 1:
                self.assertEqual(
                    decimal.Decimal(output_str),
                    decimal.Decimal(best_str),
                )

    def _comparison_test_values(self):
        zeros = [Float16('0.0'), Float16('-0.0'), 0]
        # List of lists;  all values in each of the inner lists are
        # equal; outer list ordered numerically.
        positives = [
            [0.4999999999],
            [Float16(0.5), 0.5],
            [Float16(1 - 2**-11), 1.0 - 2.0**-11],
            [1.0 - 2.0**-12],
            [Float16('1.0'), 1, 1.0],
            [1.0 + 2.0 ** -11],
            [Float16(1 + 2**-10), 1 + 2.0**-10],
            [Float16(1.5)],
            [Float16('2.0'), 2],
            [Float16(2**11-1), 2**11 - 1],
            [Float16(2**11), 2**11],
            [2**11 + 1],
            [Float16(2**16 - 2**5), 2**16 - 2**5],
            [2**16 - 2**4 - 1],
            [2**16 - 2**4],
            [2**16],
            [Float16('inf')],
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
        nans = [Float16('nan'), Float16('-nan(123)')]
        for xset in all_pairs:
            for x in xset:
                for y in nans:
                    yield x, y, 'UN'
                    yield y, x, 'UN'
        for x in nans:
            for y in nans:
                yield x, y, 'UN'

        # signaling nans
        snans = [Float16('-snan'), Float16('snan(456)')]
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
        for x, y, reln in self._comparison_test_values():
            if reln == 'EQ':
                self.assertTrue(x == y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x != y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x < y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x > y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertTrue(x <= y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertTrue(x >= y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertEqual(hash(x), hash(y))
            elif reln == 'LT':
                self.assertFalse(x == y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertTrue(x != y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertTrue(x < y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x > y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertTrue(x <= y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x >= y, msg='{!r} {!r} {!r}'.format(x, y, reln))
            elif reln == 'GT':
                self.assertFalse(x == y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertTrue(x != y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x < y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertTrue(x > y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x <= y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertTrue(x >= y, msg='{!r} {!r} {!r}'.format(x, y, reln))
            elif reln == 'UN':
                self.assertFalse(x == y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertTrue(x != y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x < y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x > y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x <= y, msg='{!r} {!r} {!r}'.format(x, y, reln))
                self.assertFalse(x >= y, msg='{!r} {!r} {!r}'.format(x, y, reln))
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
            self.assertEqual(hash(Float16(test_string)), hash(float(test_string)))

        # Signaling NaNs can't be hashed.
        snan = Float16('snan')
        with self.assertRaises(ValueError):
            hash(snan)


if __name__ == '__main__':
    unittest.main()
