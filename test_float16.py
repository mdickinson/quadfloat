import decimal
import unittest

from binary_interchange_format import BinaryInterchangeFormat
from binary_interchange_format import rounding_direction
from binary_interchange_format import (
    round_ties_to_even,
    round_ties_to_away,
    round_toward_zero,
    round_toward_positive,
    round_toward_negative,
)

from binary_interchange_format import (
    inexact_handler,
    invalid_operation_handler,
)

from binary_interchange_format import _bytes_from_iterable, _divide_nearest


float16 = BinaryInterchangeFormat(width=16)
float32 = BinaryInterchangeFormat(width=32)
float64 = BinaryInterchangeFormat(width=64)


# float16 details:
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
        Assert that two float16 instances are interchangeable.

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
        self.assertEqual(float16(2048).encode(), b'\x00\x68')
        self.assertEqual(float16(2049).encode(), b'\x00\x68')
        self.assertEqual(float16(2050).encode(), b'\x01\x68')
        self.assertEqual(float16(2051).encode(), b'\x02\x68')
        self.assertEqual(float16(2052).encode(), b'\x02\x68')
        self.assertEqual(float16(2053).encode(), b'\x02\x68')
        self.assertEqual(float16(2054).encode(), b'\x03\x68')
        self.assertEqual(float16(2055).encode(), b'\x04\x68')
        self.assertEqual(float16(2056).encode(), b'\x04\x68')

    def test_construction_from_float(self):
        self.assertInterchangeable(float16(0.9), float16('0.89990234375'))

        # Test round-half-to-even

        self.assertEqual(float16(2048.0).encode(), b'\x00\x68')
        # halfway case
        self.assertEqual(float16(2048.9999999999).encode(), b'\x00\x68')
        self.assertEqual(float16(2049.0).encode(), b'\x00\x68')
        self.assertEqual(float16(2049.0000000001).encode(), b'\x01\x68')
        self.assertEqual(float16(2050.0).encode(), b'\x01\x68')
        self.assertEqual(float16(2050.9999999999).encode(), b'\x01\x68')
        self.assertEqual(float16(2051.0).encode(), b'\x02\x68')
        self.assertEqual(float16(2051.0000000001).encode(), b'\x02\x68')
        self.assertEqual(float16(2052.0).encode(), b'\x02\x68')
        self.assertEqual(float16(2053.0).encode(), b'\x02\x68')
        self.assertEqual(float16(2054.0).encode(), b'\x03\x68')
        self.assertEqual(float16(2055.0).encode(), b'\x04\x68')
        self.assertEqual(float16(2056.0).encode(), b'\x04\x68')

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
            self.assertEqual(float16(x).encode(), bs)

    def test_division(self):
        # Test division, with particular attention to correct rounding.
        # Integers up to 2048 all representable in float16

        self.assertInterchangeable(
            float16.division(float32(2048*4), float32(4)),
            float16(2048)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 1), float32(4)),
            float16(2048)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 2), float32(4)),
            float16(2048)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 3), float32(4)),
            float16(2048)
        )
        # Exact halfway case; should be rounded down.
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 4), float32(4)),
            float16(2048)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 5), float32(4)),
            float16(2050)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 6), float32(4)),
            float16(2050)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 7), float32(4)),
            float16(2050)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 8), float32(4)),
            float16(2050)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 9), float32(4)),
            float16(2050)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 10), float32(4)),
            float16(2050)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 11), float32(4)),
            float16(2050)
        )
        # Exact halfway case, rounds *up*!
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 12), float32(4)),
            float16(2052)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 13), float32(4)),
            float16(2052)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 14), float32(4)),
            float16(2052)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 15), float32(4)),
            float16(2052)
        )
        self.assertInterchangeable(
            float16.division(float32(2048*4 + 16), float32(4)),
            float16(2052)
        )

    def test_sqrt(self):
        # Easy small integer cases.
        for i in range(1, 46):
            self.assertInterchangeable(
                float16.square_root(float16(i * i)),
                float16(i),
            )

        # Zeros.
        self.assertInterchangeable(
            float16.square_root(float16('0')),
            float16('0'),
        )
        self.assertInterchangeable(
            float16.square_root(float16('-0')),
            float16('-0'),
        )

        # Infinities
        self.assertInterchangeable(
            float16.square_root(float16('inf')),
            float16('inf'),
        )

        self.assertInterchangeable(
            float16.square_root(float16('-inf')),
            float16('nan'),
        )

        # Negatives
        self.assertInterchangeable(
            float16.square_root(float16('-4.0')),
            float16('nan'),
        )

        # NaNs
        self.assertInterchangeable(
            float16.square_root(float16('snan(456)')),
            float16('nan(456)'),
        )

        self.assertInterchangeable(
            float16.square_root(float16('-nan(123)')),
            float16('-nan(123)'),
        )

        # Subnormal results.
        tiny = 2.0**-24  # smallest positive representable float16 subnormal

        self.assertInterchangeable(
            float16.square_root(float64(tiny * tiny)),
            float16(tiny),
        )
        self.assertInterchangeable(
            float16.square_root(float64(tiny * tiny * 0.25)),
            float16('0.0'),
        )
        self.assertInterchangeable(
            float16.square_root(float64(tiny * tiny * 0.250000001)),
            float16(tiny),
        )
        self.assertInterchangeable(
            float16.square_root(float64(tiny * tiny)),
            float16(tiny),
        )
        self.assertInterchangeable(
            float16.square_root(float64(tiny * tiny * 2.24999999999)),
            float16(tiny),
        )
        self.assertInterchangeable(
            float16.square_root(float64(tiny * tiny * 2.25)),
            float16(2 * tiny),
        )
        self.assertInterchangeable(
            float16.square_root(float64(tiny * tiny * 2.250000001)),
            float16(2 * tiny),
        )
        self.assertInterchangeable(
            float16.square_root(float64(tiny * tiny * 4.0)),
            float16(2 * tiny),
        )

    def test_repr_construct_roundtrip(self):
        # Roundtrip tests.

        # Particularly interesting values.
        test_values = [
            float16('4152'),  # sits at the middle of a *closed* interval with one endpoint at 4150;
                              # so '415e1' is an acceptable short representation.
            float16('4148'),  # sits at the middle of an *open* interval with one endpoint at 4150;
                              # so '415e1' is *not* an acceptable short representation.
            float16('0.0078125'),  # power of 2;  interval needs special casing.
            float16('0.015625')  # another power of 2 where rounding to nearest for the best
                                # final digit produces a value out of range.
        ]

        # With float16, it's feasible to test *all* the values.
        for high_byte in range(256):
            for low_byte in range(256):
                value = float16.decode(_bytes_from_iterable([low_byte, high_byte]))
                test_values.append(value)

        for value in test_values:
            repr_value = repr(value)
            reconstructed_value = eval(repr_value)
            self.assertInterchangeable(value, reconstructed_value)

            str_value = str(value)
            reconstructed_value = float16(str_value)
            self.assertInterchangeable(value, reconstructed_value)

    def test_short_float_repr(self):
        # Test that we're giving back the *shortest* representation.

        # First, just tests for the value represented by the output string,
        # rather than the string representation itself (exact positioning of
        # decimal point, exponent, etc.)

        TINY = 2.0**-24
        test_pairs = [
            (float16(TINY), '6e-8'),
            (float16(2 * TINY), '1e-7'),
            (float16(3 * TINY), '2e-7'),
            (float16(4 * TINY), '2.4e-7'),
            (float16(5 * TINY), '3e-7'),
            (float16('0.015625'), '0.01563'),
            (float16('1.23'), '1.23'),
            (float16('4152'), '415e1'),
            (float16('4148'), '4148'),
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
            output_string = str(float16(input_string))
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
        zeros = [float16('0.0'), float16('-0.0'), 0]
        # List of lists;  all values in each of the inner lists are
        # equal; outer list ordered numerically.
        positives = [
            [0.4999999999],
            [float16(0.5), 0.5],
            [float16(1 - 2**-11), 1.0 - 2.0**-11],
            [1.0 - 2.0**-12],
            [float64(1 - 2.0**-53), 1 - 2.0**-53],
            [float16('1.0'), 1, 1.0],
            [float64(1 + 2.0**-52), 1 + 2.0**-52],
            [1.0 + 2.0 ** -11],
            [float16(1 + 2**-10), 1 + 2.0**-10],
            [float16(1.5)],
            [float16('2.0'), 2],
            [float16(2**11-1), 2**11 - 1],
            [float16(2**11), 2**11],
            [2**11 + 1],
            [float16(2**16 - 2**5), 2**16 - 2**5],
            [2**16 - 2**4 - 1],
            [2**16 - 2**4],
            [2**16],
            [float16('inf')],
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
        nans = [float16('nan'), float16('-nan(123)')]
        for xset in all_pairs:
            for x in xset:
                for y in nans:
                    yield x, y, 'UN'
                    yield y, x, 'UN'
        for x in nans:
            for y in nans:
                yield x, y, 'UN'

        # signaling nans
        snans = [float16('-snan'), float16('snan(456)')]
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
            self.assertEqual(hash(float16(test_string)), hash(float(test_string)))

        # Signaling NaNs can't be hashed.
        snan = float16('snan')
        with self.assertRaises(ValueError):
            hash(snan)

    def test_round_to_integral_ties_to_even(self):
        test_values = [
            (float16('inf'), float16('inf')),
            (float16('-inf'), float16('-inf')),
            (float16('-0.51'), float16('-1.0')),
            (float16('-0.5'), float16('-0.0')),
            (float16('-0.49'), float16('-0.0')),
            (float16(-2.0**-24), float16('-0.0')),
            (float16('-0.0'), float16('-0.0')),
            (float16('0.0'), float16('0.0')),
            (float16(2.0**-24), float16('0.0')),
            (float16('0.49'), float16('0.0')),
            (float16('0.5'), float16('0.0')),
            (float16('0.51'), float16('1.0')),
            (float16('0.99'), float16('1.0')),
            (float16('1.0'), float16('1.0')),
            (float16('1.49'), float16('1.0')),
            (float16('1.5'), float16('2.0')),
            (float16(2**9 - 2.0), float16(2**9 - 2)),
            (float16(2**9 - 1.75), float16(2**9 - 2)),
            (float16(2**9 - 1.5), float16(2**9 - 2)),
            (float16(2**9 - 1.25), float16(2**9 - 1)),
            (float16(2**9 - 1.0), float16(2**9 - 1)),
            (float16(2**9 - 0.75), float16(2**9 - 1)),
            (float16(2**9 - 0.5), float16(2**9)),
            (float16(2**9 - 0.25), float16(2**9)),
            (float16(2**9), float16(2**9)),
            (float16(2**9 + 0.5), float16(2**9)),
            (float16(2**9 + 1), float16(2**9 + 1)),
            (float16(2**9 + 1.5), float16(2**9 + 2)),
            (float16(2**9 + 2), float16(2**9 + 2)),
            (float16(2**9 + 2.5), float16(2**9 + 2)),
            (float16(2**9 + 3), float16(2**9 + 3)),
            (float16(2**9 + 3.5), float16(2**9 + 4)),
            (float16(2**9 + 4), float16(2**9 + 4)),
            (float16(2**10 - 4), float16(2**10 - 4)),
            (float16(2**10 - 3.5), float16(2**10 - 4)),
            (float16(2**10 - 3.0), float16(2**10 - 3)),
            (float16(2**10 - 2.5), float16(2**10 - 2)),
            (float16(2**10 - 2.0), float16(2**10 - 2)),
            (float16(2**10 - 1.5), float16(2**10 - 2)),
            (float16(2**10 - 1), float16(2**10 - 1)),
            (float16(2**10 - 0.5), float16(2**10)),
            (float16(2**10), float16(2**10)),
            (float16(2**11-1), float16(2**11-1)),
            (float16(2**11), float16(2**11)),
            (float16(2**11+2), float16(2**11+2)),
            (float16('nan'), float16('nan')),
            (float16('-nan(123)'), float16('-nan(123)')),
        ]

        for input, expected in test_values:
            signal_list = []
            with inexact_handler(signal_list.append):
                actual = input.round_to_integral_ties_to_even()
            self.assertInterchangeable(actual, expected)
            self.assertEqual(len(signal_list), 0)

        # Signaling nans should signal the invalid operation exception.
        # (And return... what?)
        signal_list = []
        input = float16('snan')
        with invalid_operation_handler(signal_list.append):
            input.round_to_integral_ties_to_even()
        self.assertEqual(len(signal_list), 1)

        # Quiet nans should *not* signal.
        signal_list = []
        input = float16('-nan(234)')
        with invalid_operation_handler(signal_list.append):
            actual = input.round_to_integral_ties_to_even()
        self.assertInterchangeable(actual, input)
        self.assertEqual(len(signal_list), 0)

    def test_round_to_integral_ties_to_away(self):
        test_values = [
            (float16('inf'), float16('inf')),
            (float16('-inf'), float16('-inf')),
            (float16('-0.51'), float16('-1.0')),
            (float16('-0.5'), float16('-1.0')),
            (float16('-0.49'), float16('-0.0')),
            (float16(-2.0**-24), float16('-0.0')),
            (float16('-0.0'), float16('-0.0')),
            (float16('0.0'), float16('0.0')),
            (float16(2.0**-24), float16('0.0')),
            (float16('0.49'), float16('0.0')),
            (float16('0.5'), float16('1.0')),
            (float16('0.51'), float16('1.0')),
            (float16('0.99'), float16('1.0')),
            (float16('1.0'), float16('1.0')),
            (float16('1.49'), float16('1.0')),
            (float16('1.5'), float16('2.0')),
            (float16(2**9 - 2.0), float16(2**9 - 2)),
            (float16(2**9 - 1.75), float16(2**9 - 2)),
            (float16(2**9 - 1.5), float16(2**9 - 1)),
            (float16(2**9 - 1.25), float16(2**9 - 1)),
            (float16(2**9 - 1.0), float16(2**9 - 1)),
            (float16(2**9 - 0.75), float16(2**9 - 1)),
            (float16(2**9 - 0.5), float16(2**9)),
            (float16(2**9 - 0.25), float16(2**9)),
            (float16(2**9), float16(2**9)),
            (float16(2**9 + 0.5), float16(2**9 + 1)),
            (float16(2**9 + 1), float16(2**9 + 1)),
            (float16(2**9 + 1.5), float16(2**9 + 2)),
            (float16(2**9 + 2), float16(2**9 + 2)),
            (float16(2**9 + 2.5), float16(2**9 + 3)),
            (float16(2**9 + 3), float16(2**9 + 3)),
            (float16(2**9 + 3.5), float16(2**9 + 4)),
            (float16(2**9 + 4), float16(2**9 + 4)),
            (float16(2**10 - 4), float16(2**10 - 4)),
            (float16(2**10 - 3.5), float16(2**10 - 3)),
            (float16(2**10 - 3.0), float16(2**10 - 3)),
            (float16(2**10 - 2.5), float16(2**10 - 2)),
            (float16(2**10 - 2.0), float16(2**10 - 2)),
            (float16(2**10 - 1.5), float16(2**10 - 1)),
            (float16(2**10 - 1), float16(2**10 - 1)),
            (float16(2**10 - 0.5), float16(2**10)),
            (float16(2**10), float16(2**10)),
            (float16(2**11-1), float16(2**11-1)),
            (float16(2**11), float16(2**11)),
            (float16(2**11+2), float16(2**11+2)),
            (float16('nan'), float16('nan')),
            (float16('-nan(123)'), float16('-nan(123)')),
        ]

        for input, expected in test_values:
            signal_list = []
            with inexact_handler(signal_list.append):
                actual = input.round_to_integral_ties_to_away()
            self.assertInterchangeable(actual, expected)
            self.assertEqual(len(signal_list), 0)

    def test_round_to_integral_toward_zero(self):
        test_values = [
            (float16('inf'), float16('inf')),
            (float16('-inf'), float16('-inf')),
            (float16('-0.51'), float16('-0.0')),
            (float16('-0.5'), float16('-0.0')),
            (float16('-0.49'), float16('-0.0')),
            (float16(-2.0**-24), float16('-0.0')),
            (float16('-0.0'), float16('-0.0')),
            (float16('0.0'), float16('0.0')),
            (float16(2.0**-24), float16('0.0')),
            (float16('0.49'), float16('0.0')),
            (float16('0.5'), float16('0.0')),
            (float16('0.51'), float16('0.0')),
            (float16('0.99'), float16('0.0')),
            (float16('1.0'), float16('1.0')),
            (float16('1.49'), float16('1.0')),
            (float16('1.5'), float16('1.0')),
            (float16(2**9 - 2.0), float16(2**9 - 2)),
            (float16(2**9 - 1.75), float16(2**9 - 2)),
            (float16(2**9 - 1.5), float16(2**9 - 2)),
            (float16(2**9 - 1.25), float16(2**9 - 2)),
            (float16(2**9 - 1.0), float16(2**9 - 1)),
            (float16(2**9 - 0.75), float16(2**9 - 1)),
            (float16(2**9 - 0.5), float16(2**9 - 1)),
            (float16(2**9 - 0.25), float16(2**9 - 1)),
            (float16(2**9), float16(2**9)),
            (float16(2**9 + 0.5), float16(2**9)),
            (float16(2**9 + 1), float16(2**9 + 1)),
            (float16(2**9 + 1.5), float16(2**9 + 1)),
            (float16(2**9 + 2), float16(2**9 + 2)),
            (float16(2**9 + 2.5), float16(2**9 + 2)),
            (float16(2**9 + 3), float16(2**9 + 3)),
            (float16(2**9 + 3.5), float16(2**9 + 3)),
            (float16(2**9 + 4), float16(2**9 + 4)),
            (float16(2**10 - 4), float16(2**10 - 4)),
            (float16(2**10 - 3.5), float16(2**10 - 4)),
            (float16(2**10 - 3.0), float16(2**10 - 3)),
            (float16(2**10 - 2.5), float16(2**10 - 3)),
            (float16(2**10 - 2.0), float16(2**10 - 2)),
            (float16(2**10 - 1.5), float16(2**10 - 2)),
            (float16(2**10 - 1), float16(2**10 - 1)),
            (float16(2**10 - 0.5), float16(2**10 - 1)),
            (float16(2**10), float16(2**10)),
            (float16(2**11-1), float16(2**11-1)),
            (float16(2**11), float16(2**11)),
            (float16(2**11+2), float16(2**11+2)),
            (float16('nan'), float16('nan')),
            (float16('-nan(123)'), float16('-nan(123)')),
        ]

        for input, expected in test_values:
            signal_list = []
            with inexact_handler(signal_list.append):
                actual = input.round_to_integral_toward_zero()
            self.assertInterchangeable(actual, expected)
            self.assertEqual(len(signal_list), 0)

    def test_round_to_integral_toward_positive(self):
        test_values = [
            (float16('inf'), float16('inf')),
            (float16('-inf'), float16('-inf')),
            (float16('-0.51'), float16('-0.0')),
            (float16('-0.5'), float16('-0.0')),
            (float16('-0.49'), float16('-0.0')),
            (float16(-2.0**-24), float16('-0.0')),
            (float16('-0.0'), float16('-0.0')),
            (float16('0.0'), float16('0.0')),
            (float16(2.0**-24), float16('1.0')),
            (float16('0.49'), float16('1.0')),
            (float16('0.5'), float16('1.0')),
            (float16('0.51'), float16('1.0')),
            (float16('0.99'), float16('1.0')),
            (float16('1.0'), float16('1.0')),
            (float16('1.49'), float16('2.0')),
            (float16('1.5'), float16('2.0')),
            (float16(2**9 - 2.0), float16(2**9 - 2)),
            (float16(2**9 - 1.75), float16(2**9 - 1)),
            (float16(2**9 - 1.5), float16(2**9 - 1)),
            (float16(2**9 - 1.25), float16(2**9 - 1)),
            (float16(2**9 - 1.0), float16(2**9 - 1)),
            (float16(2**9 - 0.75), float16(2**9 - 0)),
            (float16(2**9 - 0.5), float16(2**9 - 0)),
            (float16(2**9 - 0.25), float16(2**9 - 0)),
            (float16(2**9), float16(2**9)),
            (float16(2**9 + 0.5), float16(2**9 + 1)),
            (float16(2**9 + 1), float16(2**9 + 1)),
            (float16(2**9 + 1.5), float16(2**9 + 2)),
            (float16(2**9 + 2), float16(2**9 + 2)),
            (float16(2**9 + 2.5), float16(2**9 + 3)),
            (float16(2**9 + 3), float16(2**9 + 3)),
            (float16(2**9 + 3.5), float16(2**9 + 4)),
            (float16(2**9 + 4), float16(2**9 + 4)),
            (float16(2**10 - 4), float16(2**10 - 4)),
            (float16(2**10 - 3.5), float16(2**10 - 3)),
            (float16(2**10 - 3.0), float16(2**10 - 3)),
            (float16(2**10 - 2.5), float16(2**10 - 2)),
            (float16(2**10 - 2.0), float16(2**10 - 2)),
            (float16(2**10 - 1.5), float16(2**10 - 1)),
            (float16(2**10 - 1), float16(2**10 - 1)),
            (float16(2**10 - 0.5), float16(2**10)),
            (float16(2**10), float16(2**10)),
            (float16(2**11-1), float16(2**11-1)),
            (float16(2**11), float16(2**11)),
            (float16(2**11+2), float16(2**11+2)),
            (float16('nan'), float16('nan')),
            (float16('-nan(123)'), float16('-nan(123)')),
        ]

        for input, expected in test_values:
            signal_list = []
            with inexact_handler(signal_list.append):
                actual = input.round_to_integral_toward_positive()
            self.assertInterchangeable(actual, expected)
            self.assertEqual(len(signal_list), 0)

    def test_round_to_integral_toward_negative(self):
        test_values = [
            (float16('inf'), float16('inf')),
            (float16('-inf'), float16('-inf')),
            (float16('-0.51'), float16('-1.0')),
            (float16('-0.5'), float16('-1.0')),
            (float16('-0.49'), float16('-1.0')),
            (float16(-2.0**-24), float16('-1.0')),
            (float16('-0.0'), float16('-0.0')),
            (float16('0.0'), float16('0.0')),
            (float16(2.0**-24), float16('0.0')),
            (float16('0.49'), float16('0.0')),
            (float16('0.5'), float16('0.0')),
            (float16('0.51'), float16('0.0')),
            (float16('0.99'), float16('0.0')),
            (float16('1.0'), float16('1.0')),
            (float16('1.49'), float16('1.0')),
            (float16('1.5'), float16('1.0')),
            (float16(2**9 - 2.0), float16(2**9 - 2)),
            (float16(2**9 - 1.75), float16(2**9 - 2)),
            (float16(2**9 - 1.5), float16(2**9 - 2)),
            (float16(2**9 - 1.25), float16(2**9 - 2)),
            (float16(2**9 - 1.0), float16(2**9 - 1)),
            (float16(2**9 - 0.75), float16(2**9 - 1)),
            (float16(2**9 - 0.5), float16(2**9 - 1)),
            (float16(2**9 - 0.25), float16(2**9 - 1)),
            (float16(2**9), float16(2**9)),
            (float16(2**9 + 0.5), float16(2**9)),
            (float16(2**9 + 1), float16(2**9 + 1)),
            (float16(2**9 + 1.5), float16(2**9 + 1)),
            (float16(2**9 + 2), float16(2**9 + 2)),
            (float16(2**9 + 2.5), float16(2**9 + 2)),
            (float16(2**9 + 3), float16(2**9 + 3)),
            (float16(2**9 + 3.5), float16(2**9 + 3)),
            (float16(2**9 + 4), float16(2**9 + 4)),
            (float16(2**10 - 4), float16(2**10 - 4)),
            (float16(2**10 - 3.5), float16(2**10 - 4)),
            (float16(2**10 - 3.0), float16(2**10 - 3)),
            (float16(2**10 - 2.5), float16(2**10 - 3)),
            (float16(2**10 - 2.0), float16(2**10 - 2)),
            (float16(2**10 - 1.5), float16(2**10 - 2)),
            (float16(2**10 - 1), float16(2**10 - 1)),
            (float16(2**10 - 0.5), float16(2**10 - 1)),
            (float16(2**10), float16(2**10)),
            (float16(2**11-1), float16(2**11-1)),
            (float16(2**11), float16(2**11)),
            (float16(2**11+2), float16(2**11+2)),
            (float16('nan'), float16('nan')),
            (float16('-nan(123)'), float16('-nan(123)')),
        ]

        for input, expected in test_values:
            signal_list = []
            with inexact_handler(signal_list.append):
                actual = input.round_to_integral_toward_negative()
            self.assertInterchangeable(actual, expected)
            self.assertEqual(len(signal_list), 0)

    def test_round_to_integral_exact(self):
        # Round to integral exact is supposed to round according to the
        # 'applicable rounding-direction' attribute.

        test_values = [float16(n / 4.0) for n in range(100)]
        test_values.extend([-x for x in test_values])

        for x in test_values:
            with rounding_direction(round_ties_to_even):
                actual = x.round_to_integral_exact()
            expected = x.round_to_integral_ties_to_even()
            self.assertInterchangeable(actual, expected)

        for x in test_values:
            with rounding_direction(round_ties_to_away):
                actual = x.round_to_integral_exact()
            expected = x.round_to_integral_ties_to_away()
            self.assertInterchangeable(actual, expected)

        for x in test_values:
            with rounding_direction(round_toward_positive):
                actual = x.round_to_integral_exact()
            expected = x.round_to_integral_toward_positive()
            self.assertInterchangeable(actual, expected)

        for x in test_values:
            with rounding_direction(round_toward_negative):
                actual = x.round_to_integral_exact()
            expected = x.round_to_integral_toward_negative()
            self.assertInterchangeable(actual, expected)

        for x in test_values:
            with rounding_direction(round_toward_zero):
                actual = x.round_to_integral_exact()
            expected = x.round_to_integral_toward_zero()
            self.assertInterchangeable(actual, expected)

    def test_round_to_integral_exact_signals(self):
        # Should signal the 'inexact' exception for inexact results.
        signal_list = []
        x = float16('1.5')
        with inexact_handler(signal_list.append):
            x.round_to_integral_exact()
        self.assertEqual(len(signal_list), 1)

        # But not for exact results.
        signal_list = []
        x = float16('1.0')
        with inexact_handler(signal_list.append):
            x.round_to_integral_exact()
        self.assertEqual(len(signal_list), 0)

    def test_next_up_and_next_down(self):
        tiny = 2.0**-24
        test_values = [
            (float16('-inf'), float16(2**5 - 2**16)),
            (float16(-1.0-2**-10), float16('-1.0')),
            (float16('-1.0'), float16(2**-11 - 1.0)),
            (float16(-2 * tiny), float16(-tiny)),
            (float16(-tiny), float16('-0.0')),
            (float16('-0.0'), float16(tiny)),
            (float16('0.0'), float16(tiny)),
            (float16(tiny), float16(2 * tiny)),
            (float16(2 * tiny), float16(3 * tiny)),
            (float16(2**-14 - 2**-24), float16(2**-14)),
            (float16(2**-14), float16(2**-14 + 2**-24)),
            (float16(2**-13 - 2**-24), float16(2**-13)),
            (float16(2**-13), float16(2**-13 + 2**-23)),
            (float16(2**16 - 2**5), float16('inf')),
            (float16(1.0 - 2**-11), float16('1.0')),
            (float16('1.0'), float16(1.0 + 2**-10)),
            (float16('inf'), float16('inf')),
            (float16('nan'), float16('nan')),
            (float16('-nan(123)'), float16('-nan(123)')),
        ]
        for input, expected in test_values:
            actual = input.next_up()
            self.assertInterchangeable(actual, expected)

            input, expected = -input, -expected
            actual = input.next_down()
            self.assertInterchangeable(actual, expected)

    def test_remainder(self):
        # The two arguments to 'remainder' should have the same type.
        x = float16('2.3')
        y = float32('1.0')
        with self.assertRaises(ValueError):
            x.remainder(y)

        test_triples = [
            # Positive second argument.
            ('-26', '13', '-0'),
            ('-20', '13', '6'),
            ('-19.5', '13', '6.5'),
            ('-19', '13', '-6'),
            ('-13', '13', '-0'),
            ('-7', '13', '6'),
            ('-6.5', '13', '-6.5'),
            ('-6', '13', '-6'),
            ('-0', '13', '-0'),
            ('0', '13', '0'),
            ('6', '13', '6'),
            ('6.5', '13', '6.5'),
            ('7', '13', '-6'),
            ('13', '13', '0'),
            ('19', '13', '6'),
            ('19.5', '13', '-6.5'),
            ('20', '13', '-6'),
            ('26', '13', '0'),

            # Negative second argument.
            ('-26', '-13', '-0'),
            ('-20', '-13', '6'),
            ('-19.5', '-13', '6.5'),
            ('-19', '-13', '-6'),
            ('-13', '-13', '-0'),
            ('-7', '-13', '6'),
            ('-6.5', '-13', '-6.5'),
            ('-6', '-13', '-6'),
            ('-0', '-13', '-0'),
            ('0', '-13', '0'),
            ('6', '-13', '6'),
            ('6.5', '-13', '6.5'),
            ('7', '-13', '-6'),
            ('13', '-13', '0'),
            ('19', '-13', '6'),
            ('19.5', '-13', '-6.5'),
            ('20', '-13', '-6'),
            ('26', '-13', '0'),

            # Additional tests.
            ('2.25', '1.0', '0.25'),
            ('2.25', '1.5', '-0.75'),
            ('65504', '7', '-2'),
            ('0.01000213623046875', '2345', '0.01000213623046875'),

            # Case where both are subnormal.
            ('-3e-7', '1e-7', '-6e-8'),   # (5 * tiny, 2 * tiny, tiny)
            ('-2.4e-7', '1e-7', '-0'),    # (4 * tiny, 2 * tiny, 0)
            ('-2e-7', '1e-7', '6e-8'),  # (3 * tiny, 2 * tiny, -tiny)
            ('-1e-7', '1e-7', '-0'),      # (2 * tiny, 2 * tiny, 0)
            ('-6e-8', '1e-7', '-6e-8'),   # (tiny, 2 * tiny, tiny)
            ('-0', '1e-7', '-0'),         # (0, 2 * tiny, 0)
            ('0', '1e-7', '0'),         # (0, 2 * tiny, 0)
            ('6e-8', '1e-7', '6e-8'),   # (tiny, 2 * tiny, tiny)
            ('1e-7', '1e-7', '0'),      # (2 * tiny, 2 * tiny, 0)
            ('2e-7', '1e-7', '-6e-8'),  # (3 * tiny, 2 * tiny, -tiny)
            ('2.4e-7', '1e-7', '0'),    # (4 * tiny, 2 * tiny, 0)
            ('3e-7', '1e-7', '6e-8'),   # (5 * tiny, 2 * tiny, tiny)

            ('0', '2e-7', '0'),         # (0, 3 * tiny, 0)
            ('6e-8', '2e-7', '6e-8'),   # (tiny, 3 * tiny, tiny)
            ('1e-7', '2e-7', '-6e-8'),  # (2 * tiny, 3 * tiny, -tiny)
            ('2e-7', '2e-7', '0'),      # (3 * tiny, 3 * tiny, 0)
            ('2.4e-7', '2e-7', '6e-8'), # (4 * tiny, 3 * tiny, tiny)
            ('3e-7', '2e-7', '-6e-8'),  # (5 * tiny, 3 * tiny, -tiny)

            # Special case: second argument is infinite.
            ('3.5', 'inf', '3.5'),
            ('-0.01', 'inf', '-0.01'),
            ('3.5', '-inf', '3.5'),
            ('-0.2', '-inf', '-0.2'),
            ('-0.0', 'inf', '-0.0'),
            ('0.0', 'inf', '0.0'),
            ('-0.0', '-inf', '-0.0'),
            ('0.0', '-inf', '0.0'),
        ]

        for source1, source2, expected in test_triples:
            source1 = float16(source1)
            source2 = float16(source2)
            expected = float16(expected)
            actual = source1.remainder(source2)
            self.assertInterchangeable(actual, expected)

        # Pairs x, y such that x.remainder(y) is invalid.
        invalid_pairs = [
            # Second argument 0.
            ('3.4', '0.0'),
            ('1.2', '-0.0'),
            ('0.0', '0.0'),
            ('0.0', '-0.0'),
            ('-0.0', '0.0'),
            ('-0.0', '-0.0'),

            # First argument infinite.
            ('inf', '2.0'),
            ('-inf', '0.5'),
            ('inf', 'inf'),
            ('inf', '-inf'),
            ('-inf', 'inf'),
            ('-inf', '-inf'),
            ('inf', '0'),
            ('inf', '-0'),
            ('-inf', '0'),
            ('-inf', '-0'),
        ]

        for source1, source2 in invalid_pairs:
            source1 = float16(source1)
            source2 = float16(source2)
            signal_list = []
            with invalid_operation_handler(signal_list.append):
                source1.remainder(source2)
            self.assertEqual(len(signal_list), 1)

    def test_min_num(self):
        test_triples = [
            # In case of equal numbers, first one wins.
            ('-0.0', '0.0', '-0.0'),
            ('0.0', '-0.0', '0.0'),
            # Infinities.
            ('-inf', '-inf', '-inf'),
            ('-inf', 'inf', '-inf'),
            ('inf', '-inf', '-inf'),
            ('inf', 'inf', 'inf'),
            ('inf', '2.3', '2.3'),
            ('-inf', '2.3', '-inf'),
            ('2.3', 'inf', '2.3'),
            ('2.3', '-inf', '-inf'),
            ('-1', '1', '-1'),
            ('1', '-1', '-1'),
            ('1.2', '1.3', '1.2'),
            ('-1.2', '-1.3', '-1.3'),
            ('0.1', '10.0', '0.1'),
            ('-10.0', '-0.1', '-10.0'),
            # Quiet NaNs
            ('1.2', 'nan(123)', '1.2'),
            ('nan(123)', '1.2', '1.2'),
            ('nan(123)', 'nan(456)', 'nan(123)'),
            ('nan(456)', 'nan(123)', 'nan(456)'),
        ]

        for source1, source2, expected in test_triples:
            source1 = float16(source1)
            source2 = float16(source2)
            expected = float16(expected)
            actual = source1.min_num(source2)
            self.assertInterchangeable(actual, expected)

    def test_max_num(self):
        test_triples = [
            # In case of equal numbers, second one wins.
            ('-0.0', '0.0', '0.0'),
            ('0.0', '-0.0', '-0.0'),
            # Infinities.
            ('-inf', '-inf', '-inf'),
            ('-inf', 'inf', 'inf'),
            ('inf', '-inf', 'inf'),
            ('inf', 'inf', 'inf'),
            ('inf', '2.3', 'inf'),
            ('-inf', '2.3', '2.3'),
            ('2.3', 'inf', 'inf'),
            ('2.3', '-inf', '2.3'),
            ('-1', '1', '1'),
            ('1', '-1', '1'),
            ('1.2', '1.3', '1.3'),
            ('-1.2', '-1.3', '-1.2'),
            ('0.1', '10.0', '10.0'),
            ('-10.0', '-0.1', '-0.1'),
            # Quiet NaNs
            ('1.2', 'nan(123)', '1.2'),
            ('nan(123)', '1.2', '1.2'),
            ('nan(123)', 'nan(456)', 'nan(123)'),
            ('nan(456)', 'nan(123)', 'nan(456)'),
        ]

        for source1, source2, expected in test_triples:
            source1 = float16(source1)
            source2 = float16(source2)
            expected = float16(expected)
            actual = source1.max_num(source2)
            self.assertInterchangeable(actual, expected)



if __name__ == '__main__':
    unittest.main()
