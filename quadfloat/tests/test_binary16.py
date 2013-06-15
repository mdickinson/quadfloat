import contextlib
import decimal

from quadfloat import binary16, binary32, binary64
from quadfloat.arithmetic import _divide_nearest
from quadfloat.attributes import (
    inexact_handler,
    invalid_operation_handler,
    rounding_direction,
)
from quadfloat.binary_interchange_format import (
    BinaryInterchangeFormat,
    round_ties_to_even,
    round_ties_to_away,
    round_toward_zero,
    round_toward_positive,
    round_toward_negative,
)
from quadfloat.compat import _bytes_from_iterable
from quadfloat.tests.base_test_case import BaseTestCase


# binary16 details:
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
    @contextlib.contextmanager
    def assertSignalsInvalidOperation(self):
        signal_list = []
        with invalid_operation_handler(signal_list.append):
            yield
        self.assertEqual(len(signal_list), 1)

    def test_construction_from_int(self):
        # Test round-half-to-even
        self.assertEqual(binary16(2048).encode(), b'\x00\x68')
        self.assertEqual(binary16(2049).encode(), b'\x00\x68')
        self.assertEqual(binary16(2050).encode(), b'\x01\x68')
        self.assertEqual(binary16(2051).encode(), b'\x02\x68')
        self.assertEqual(binary16(2052).encode(), b'\x02\x68')
        self.assertEqual(binary16(2053).encode(), b'\x02\x68')
        self.assertEqual(binary16(2054).encode(), b'\x03\x68')
        self.assertEqual(binary16(2055).encode(), b'\x04\x68')
        self.assertEqual(binary16(2056).encode(), b'\x04\x68')

    def test_construction_from_float(self):
        self.assertInterchangeable(binary16(0.9), binary16('0.89990234375'))

        # Test round-half-to-even
        self.assertEqual(binary16(2048.0).encode(), b'\x00\x68')
        self.assertEqual(binary16(2048.9999999999).encode(), b'\x00\x68')
        self.assertEqual(binary16(2049.0).encode(), b'\x00\x68')
        self.assertEqual(binary16(2049.0000000001).encode(), b'\x01\x68')
        self.assertEqual(binary16(2050.0).encode(), b'\x01\x68')
        self.assertEqual(binary16(2050.9999999999).encode(), b'\x01\x68')
        self.assertEqual(binary16(2051.0).encode(), b'\x02\x68')
        self.assertEqual(binary16(2051.0000000001).encode(), b'\x02\x68')
        self.assertEqual(binary16(2052.0).encode(), b'\x02\x68')
        self.assertEqual(binary16(2053.0).encode(), b'\x02\x68')
        self.assertEqual(binary16(2054.0).encode(), b'\x03\x68')
        self.assertEqual(binary16(2055.0).encode(), b'\x04\x68')
        self.assertEqual(binary16(2056.0).encode(), b'\x04\x68')

        # Subnormals.
        eps = 1e-10
        tiny = 2.0 ** -24  # smallest positive representable binary16 subnormal
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
            self.assertEqual(binary16(x).encode(), bs)

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
        for high_byte in range(256):
            for low_byte in range(256):
                value = binary16.decode(
                    _bytes_from_iterable([low_byte, high_byte]))
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
            (binary16('4152'), '415e1'),
            (binary16('4148'), '4148'),
        ]
        for input, output_string in test_pairs:
            input_string = str(input)
            self.assertEqual(
                decimal.Decimal(input_string),
                decimal.Decimal(output_string),
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
        # 10 ** e for some e.  For the standard formats, an interval of this
        # relative width can only occur for subnormal target values.

        # This corner case doesn't occur for binary16, binary32, binary64 or
        # binary128.

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
            best_str = '{0}e{1}'.format(best_digit, n - 1)

            # If the returned string has only one digit, it should be
            # equal to the closest string.
            output_str = str(binary_value)
            if len(decimal.Decimal(output_str)._int.strip('0')) == 1:
                self.assertEqual(
                    decimal.Decimal(output_str),
                    decimal.Decimal(best_str),
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

    def test_round_to_integral_ties_to_even(self):
        test_values = [
            (binary16('inf'), binary16('inf')),
            (binary16('-inf'), binary16('-inf')),
            (binary16('-0.51'), binary16('-1.0')),
            (binary16('-0.5'), binary16('-0.0')),
            (binary16('-0.49'), binary16('-0.0')),
            (binary16(-2.0 ** -24), binary16('-0.0')),
            (binary16('-0.0'), binary16('-0.0')),
            (binary16('0.0'), binary16('0.0')),
            (binary16(2.0 ** -24), binary16('0.0')),
            (binary16('0.49'), binary16('0.0')),
            (binary16('0.5'), binary16('0.0')),
            (binary16('0.51'), binary16('1.0')),
            (binary16('0.99'), binary16('1.0')),
            (binary16('1.0'), binary16('1.0')),
            (binary16('1.49'), binary16('1.0')),
            (binary16('1.5'), binary16('2.0')),
            (binary16(2 ** 9 - 2.0), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.75), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.5), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.25), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 1.0), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.75), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.5), binary16(2 ** 9)),
            (binary16(2 ** 9 - 0.25), binary16(2 ** 9)),
            (binary16(2 ** 9), binary16(2 ** 9)),
            (binary16(2 ** 9 + 0.5), binary16(2 ** 9)),
            (binary16(2 ** 9 + 1), binary16(2 ** 9 + 1)),
            (binary16(2 ** 9 + 1.5), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 2), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 2.5), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 3), binary16(2 ** 9 + 3)),
            (binary16(2 ** 9 + 3.5), binary16(2 ** 9 + 4)),
            (binary16(2 ** 9 + 4), binary16(2 ** 9 + 4)),
            (binary16(2 ** 10 - 4), binary16(2 ** 10 - 4)),
            (binary16(2 ** 10 - 3.5), binary16(2 ** 10 - 4)),
            (binary16(2 ** 10 - 3.0), binary16(2 ** 10 - 3)),
            (binary16(2 ** 10 - 2.5), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 2.0), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 1.5), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 1), binary16(2 ** 10 - 1)),
            (binary16(2 ** 10 - 0.5), binary16(2 ** 10)),
            (binary16(2 ** 10), binary16(2 ** 10)),
            (binary16(2 ** 11 - 1), binary16(2 ** 11 - 1)),
            (binary16(2 ** 11), binary16(2 ** 11)),
            (binary16(2 ** 11 + 2), binary16(2 ** 11 + 2)),
            (binary16('nan'), binary16('nan')),
            (binary16('-nan(123)'), binary16('-nan(123)')),
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
        input = binary16('snan')
        with invalid_operation_handler(signal_list.append):
            input.round_to_integral_ties_to_even()
        self.assertEqual(len(signal_list), 1)

        # Quiet nans should *not* signal.
        signal_list = []
        input = binary16('-nan(234)')
        with invalid_operation_handler(signal_list.append):
            actual = input.round_to_integral_ties_to_even()
        self.assertInterchangeable(actual, input)
        self.assertEqual(len(signal_list), 0)

    def test_round_to_integral_ties_to_away(self):
        test_values = [
            (binary16('inf'), binary16('inf')),
            (binary16('-inf'), binary16('-inf')),
            (binary16('-0.51'), binary16('-1.0')),
            (binary16('-0.5'), binary16('-1.0')),
            (binary16('-0.49'), binary16('-0.0')),
            (binary16(-2.0 ** -24), binary16('-0.0')),
            (binary16('-0.0'), binary16('-0.0')),
            (binary16('0.0'), binary16('0.0')),
            (binary16(2.0 ** -24), binary16('0.0')),
            (binary16('0.49'), binary16('0.0')),
            (binary16('0.5'), binary16('1.0')),
            (binary16('0.51'), binary16('1.0')),
            (binary16('0.99'), binary16('1.0')),
            (binary16('1.0'), binary16('1.0')),
            (binary16('1.49'), binary16('1.0')),
            (binary16('1.5'), binary16('2.0')),
            (binary16(2 ** 9 - 2.0), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.75), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.5), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 1.25), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 1.0), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.75), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.5), binary16(2 ** 9)),
            (binary16(2 ** 9 - 0.25), binary16(2 ** 9)),
            (binary16(2 ** 9), binary16(2 ** 9)),
            (binary16(2 ** 9 + 0.5), binary16(2 ** 9 + 1)),
            (binary16(2 ** 9 + 1), binary16(2 ** 9 + 1)),
            (binary16(2 ** 9 + 1.5), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 2), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 2.5), binary16(2 ** 9 + 3)),
            (binary16(2 ** 9 + 3), binary16(2 ** 9 + 3)),
            (binary16(2 ** 9 + 3.5), binary16(2 ** 9 + 4)),
            (binary16(2 ** 9 + 4), binary16(2 ** 9 + 4)),
            (binary16(2 ** 10 - 4), binary16(2 ** 10 - 4)),
            (binary16(2 ** 10 - 3.5), binary16(2 ** 10 - 3)),
            (binary16(2 ** 10 - 3.0), binary16(2 ** 10 - 3)),
            (binary16(2 ** 10 - 2.5), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 2.0), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 1.5), binary16(2 ** 10 - 1)),
            (binary16(2 ** 10 - 1), binary16(2 ** 10 - 1)),
            (binary16(2 ** 10 - 0.5), binary16(2 ** 10)),
            (binary16(2 ** 10), binary16(2 ** 10)),
            (binary16(2 ** 11 - 1), binary16(2 ** 11 - 1)),
            (binary16(2 ** 11), binary16(2 ** 11)),
            (binary16(2 ** 11 + 2), binary16(2 ** 11 + 2)),
            (binary16('nan'), binary16('nan')),
            (binary16('-nan(123)'), binary16('-nan(123)')),
        ]

        for input, expected in test_values:
            signal_list = []
            with inexact_handler(signal_list.append):
                actual = input.round_to_integral_ties_to_away()
            self.assertInterchangeable(actual, expected)
            self.assertEqual(len(signal_list), 0)

    def test_round_to_integral_toward_zero(self):
        test_values = [
            (binary16('inf'), binary16('inf')),
            (binary16('-inf'), binary16('-inf')),
            (binary16('-0.51'), binary16('-0.0')),
            (binary16('-0.5'), binary16('-0.0')),
            (binary16('-0.49'), binary16('-0.0')),
            (binary16(-2.0 ** -24), binary16('-0.0')),
            (binary16('-0.0'), binary16('-0.0')),
            (binary16('0.0'), binary16('0.0')),
            (binary16(2.0 ** -24), binary16('0.0')),
            (binary16('0.49'), binary16('0.0')),
            (binary16('0.5'), binary16('0.0')),
            (binary16('0.51'), binary16('0.0')),
            (binary16('0.99'), binary16('0.0')),
            (binary16('1.0'), binary16('1.0')),
            (binary16('1.49'), binary16('1.0')),
            (binary16('1.5'), binary16('1.0')),
            (binary16(2 ** 9 - 2.0), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.75), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.5), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.25), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.0), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.75), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.5), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.25), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9), binary16(2 ** 9)),
            (binary16(2 ** 9 + 0.5), binary16(2 ** 9)),
            (binary16(2 ** 9 + 1), binary16(2 ** 9 + 1)),
            (binary16(2 ** 9 + 1.5), binary16(2 ** 9 + 1)),
            (binary16(2 ** 9 + 2), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 2.5), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 3), binary16(2 ** 9 + 3)),
            (binary16(2 ** 9 + 3.5), binary16(2 ** 9 + 3)),
            (binary16(2 ** 9 + 4), binary16(2 ** 9 + 4)),
            (binary16(2 ** 10 - 4), binary16(2 ** 10 - 4)),
            (binary16(2 ** 10 - 3.5), binary16(2 ** 10 - 4)),
            (binary16(2 ** 10 - 3.0), binary16(2 ** 10 - 3)),
            (binary16(2 ** 10 - 2.5), binary16(2 ** 10 - 3)),
            (binary16(2 ** 10 - 2.0), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 1.5), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 1), binary16(2 ** 10 - 1)),
            (binary16(2 ** 10 - 0.5), binary16(2 ** 10 - 1)),
            (binary16(2 ** 10), binary16(2 ** 10)),
            (binary16(2 ** 11 - 1), binary16(2 ** 11 - 1)),
            (binary16(2 ** 11), binary16(2 ** 11)),
            (binary16(2 ** 11 + 2), binary16(2 ** 11 + 2)),
            (binary16('nan'), binary16('nan')),
            (binary16('-nan(123)'), binary16('-nan(123)')),
        ]

        for input, expected in test_values:
            signal_list = []
            with inexact_handler(signal_list.append):
                actual = input.round_to_integral_toward_zero()
            self.assertInterchangeable(actual, expected)
            self.assertEqual(len(signal_list), 0)

    def test_round_to_integral_toward_positive(self):
        test_values = [
            (binary16('inf'), binary16('inf')),
            (binary16('-inf'), binary16('-inf')),
            (binary16('-0.51'), binary16('-0.0')),
            (binary16('-0.5'), binary16('-0.0')),
            (binary16('-0.49'), binary16('-0.0')),
            (binary16(-2.0 ** -24), binary16('-0.0')),
            (binary16('-0.0'), binary16('-0.0')),
            (binary16('0.0'), binary16('0.0')),
            (binary16(2.0 ** -24), binary16('1.0')),
            (binary16('0.49'), binary16('1.0')),
            (binary16('0.5'), binary16('1.0')),
            (binary16('0.51'), binary16('1.0')),
            (binary16('0.99'), binary16('1.0')),
            (binary16('1.0'), binary16('1.0')),
            (binary16('1.49'), binary16('2.0')),
            (binary16('1.5'), binary16('2.0')),
            (binary16(2 ** 9 - 2.0), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.75), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 1.5), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 1.25), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 1.0), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.75), binary16(2 ** 9 - 0)),
            (binary16(2 ** 9 - 0.5), binary16(2 ** 9 - 0)),
            (binary16(2 ** 9 - 0.25), binary16(2 ** 9 - 0)),
            (binary16(2 ** 9), binary16(2 ** 9)),
            (binary16(2 ** 9 + 0.5), binary16(2 ** 9 + 1)),
            (binary16(2 ** 9 + 1), binary16(2 ** 9 + 1)),
            (binary16(2 ** 9 + 1.5), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 2), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 2.5), binary16(2 ** 9 + 3)),
            (binary16(2 ** 9 + 3), binary16(2 ** 9 + 3)),
            (binary16(2 ** 9 + 3.5), binary16(2 ** 9 + 4)),
            (binary16(2 ** 9 + 4), binary16(2 ** 9 + 4)),
            (binary16(2 ** 10 - 4), binary16(2 ** 10 - 4)),
            (binary16(2 ** 10 - 3.5), binary16(2 ** 10 - 3)),
            (binary16(2 ** 10 - 3.0), binary16(2 ** 10 - 3)),
            (binary16(2 ** 10 - 2.5), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 2.0), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 1.5), binary16(2 ** 10 - 1)),
            (binary16(2 ** 10 - 1), binary16(2 ** 10 - 1)),
            (binary16(2 ** 10 - 0.5), binary16(2 ** 10)),
            (binary16(2 ** 10), binary16(2 ** 10)),
            (binary16(2 ** 11 - 1), binary16(2 ** 11 - 1)),
            (binary16(2 ** 11), binary16(2 ** 11)),
            (binary16(2 ** 11 + 2), binary16(2 ** 11 + 2)),
            (binary16('nan'), binary16('nan')),
            (binary16('-nan(123)'), binary16('-nan(123)')),
        ]

        for input, expected in test_values:
            signal_list = []
            with inexact_handler(signal_list.append):
                actual = input.round_to_integral_toward_positive()
            self.assertInterchangeable(actual, expected)
            self.assertEqual(len(signal_list), 0)

    def test_round_to_integral_toward_negative(self):
        test_values = [
            (binary16('inf'), binary16('inf')),
            (binary16('-inf'), binary16('-inf')),
            (binary16('-0.51'), binary16('-1.0')),
            (binary16('-0.5'), binary16('-1.0')),
            (binary16('-0.49'), binary16('-1.0')),
            (binary16(-2.0 ** -24), binary16('-1.0')),
            (binary16('-0.0'), binary16('-0.0')),
            (binary16('0.0'), binary16('0.0')),
            (binary16(2.0 ** -24), binary16('0.0')),
            (binary16('0.49'), binary16('0.0')),
            (binary16('0.5'), binary16('0.0')),
            (binary16('0.51'), binary16('0.0')),
            (binary16('0.99'), binary16('0.0')),
            (binary16('1.0'), binary16('1.0')),
            (binary16('1.49'), binary16('1.0')),
            (binary16('1.5'), binary16('1.0')),
            (binary16(2 ** 9 - 2.0), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.75), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.5), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.25), binary16(2 ** 9 - 2)),
            (binary16(2 ** 9 - 1.0), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.75), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.5), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9 - 0.25), binary16(2 ** 9 - 1)),
            (binary16(2 ** 9), binary16(2 ** 9)),
            (binary16(2 ** 9 + 0.5), binary16(2 ** 9)),
            (binary16(2 ** 9 + 1), binary16(2 ** 9 + 1)),
            (binary16(2 ** 9 + 1.5), binary16(2 ** 9 + 1)),
            (binary16(2 ** 9 + 2), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 2.5), binary16(2 ** 9 + 2)),
            (binary16(2 ** 9 + 3), binary16(2 ** 9 + 3)),
            (binary16(2 ** 9 + 3.5), binary16(2 ** 9 + 3)),
            (binary16(2 ** 9 + 4), binary16(2 ** 9 + 4)),
            (binary16(2 ** 10 - 4), binary16(2 ** 10 - 4)),
            (binary16(2 ** 10 - 3.5), binary16(2 ** 10 - 4)),
            (binary16(2 ** 10 - 3.0), binary16(2 ** 10 - 3)),
            (binary16(2 ** 10 - 2.5), binary16(2 ** 10 - 3)),
            (binary16(2 ** 10 - 2.0), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 1.5), binary16(2 ** 10 - 2)),
            (binary16(2 ** 10 - 1), binary16(2 ** 10 - 1)),
            (binary16(2 ** 10 - 0.5), binary16(2 ** 10 - 1)),
            (binary16(2 ** 10), binary16(2 ** 10)),
            (binary16(2 ** 11 - 1), binary16(2 ** 11 - 1)),
            (binary16(2 ** 11), binary16(2 ** 11)),
            (binary16(2 ** 11 + 2), binary16(2 ** 11 + 2)),
            (binary16('nan'), binary16('nan')),
            (binary16('-nan(123)'), binary16('-nan(123)')),
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

        test_values = [binary16(n / 4.0) for n in range(100)]
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
        x = binary16('1.5')
        with inexact_handler(signal_list.append):
            x.round_to_integral_exact()
        self.assertEqual(len(signal_list), 1)

        # But not for exact results.
        signal_list = []
        x = binary16('1.0')
        with inexact_handler(signal_list.append):
            x.round_to_integral_exact()
        self.assertEqual(len(signal_list), 0)

    def test_next_up_and_next_down(self):
        tiny = 2.0 ** -24
        test_values = [
            (binary16('-inf'), binary16(2 ** 5 - 2 ** 16)),
            (binary16(-1.0 - 2 ** -10), binary16('-1.0')),
            (binary16('-1.0'), binary16(2 ** -11 - 1.0)),
            (binary16(-2 * tiny), binary16(-tiny)),
            (binary16(-tiny), binary16('-0.0')),
            (binary16('-0.0'), binary16(tiny)),
            (binary16('0.0'), binary16(tiny)),
            (binary16(tiny), binary16(2 * tiny)),
            (binary16(2 * tiny), binary16(3 * tiny)),
            (binary16(2 ** -14 - 2 ** -24), binary16(2 ** -14)),
            (binary16(2 ** -14), binary16(2 ** -14 + 2 ** -24)),
            (binary16(2 ** -13 - 2 ** -24), binary16(2 ** -13)),
            (binary16(2 ** -13), binary16(2 ** -13 + 2 ** -23)),
            (binary16(2 ** 16 - 2 ** 5), binary16('inf')),
            (binary16(1.0 - 2 ** -11), binary16('1.0')),
            (binary16('1.0'), binary16(1.0 + 2 ** -10)),
            (binary16('inf'), binary16('inf')),
            (binary16('nan'), binary16('nan')),
            (binary16('-nan(123)'), binary16('-nan(123)')),
        ]
        for input, expected in test_values:
            actual = input.next_up()
            self.assertInterchangeable(actual, expected)

            input, expected = -input, -expected
            actual = input.next_down()
            self.assertInterchangeable(actual, expected)

    def test_remainder(self):
        # The two arguments to 'remainder' should have the same type.
        x = binary16('2.3')
        y = binary32('1.0')
        with self.assertRaises(ValueError):
            x.remainder(y)

        # Signaling NaNs are tested in remainder.qtest.
        test_triples = [
            # Quiet NaNs
            ('nan(123)', 'nan(456)', 'nan(123)'),
            ('nan(345)', '23.4', 'nan(345)'),
            ('16.4', '-nan(789)', '-nan(789)'),

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
            ('2.4e-7', '2e-7', '6e-8'),  # (4 * tiny, 3 * tiny, tiny)
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
            source1 = binary16(source1)
            source2 = binary16(source2)
            expected = binary16(expected)
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
            source1 = binary16(source1)
            source2 = binary16(source2)
            signal_list = []
            with invalid_operation_handler(signal_list.append):
                source1.remainder(source2)
            self.assertEqual(len(signal_list), 1)

    def test_min_num(self):
        x = binary16('2.3')
        y = binary32('1.0')
        with self.assertRaises(ValueError):
            x.min_num(y)

        test_triples = [
            # In case of equal numbers, compare signs.
            ('-0.0', '0.0', '-0.0'),
            ('0.0', '-0.0', '-0.0'),
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
            source1 = binary16(source1)
            source2 = binary16(source2)
            expected = binary16(expected)
            actual = source1.min_num(source2)
            self.assertInterchangeable(actual, expected)

    def test_max_num(self):
        x = binary16('2.3')
        y = binary32('1.0')
        with self.assertRaises(ValueError):
            x.max_num(y)

        test_triples = [
            # In case of equal numbers, compare signs.
            ('-0.0', '0.0', '0.0'),
            ('0.0', '-0.0', '0.0'),
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
            source1 = binary16(source1)
            source2 = binary16(source2)
            expected = binary16(expected)
            actual = source1.max_num(source2)
            self.assertInterchangeable(actual, expected)

    def test_min_num_mag(self):
        x = binary16('2.3')
        y = binary32('1.0')
        with self.assertRaises(ValueError):
            x.min_num_mag(y)

        test_triples = [
            # In case of equal numbers, compare signs.
            ('-0.0', '0.0', '-0.0'),
            ('0.0', '-0.0', '-0.0'),
            # Infinities.
            ('-inf', '-inf', '-inf'),
            ('-inf', 'inf', '-inf'),
            ('inf', '-inf', '-inf'),
            ('inf', 'inf', 'inf'),
            ('inf', '2.3', '2.3'),
            ('-inf', '2.3', '2.3'),
            ('2.3', 'inf', '2.3'),
            ('2.3', '-inf', '2.3'),
            ('-1', '1', '-1'),
            ('1', '-1', '-1'),
            ('1.2', '1.3', '1.2'),
            ('-1.2', '-1.3', '-1.2'),
            ('0.1', '10.0', '0.1'),
            ('-10.0', '-0.1', '-0.1'),
            # Quiet NaNs
            ('1.2', 'nan(123)', '1.2'),
            ('nan(123)', '1.2', '1.2'),
            ('nan(123)', 'nan(456)', 'nan(123)'),
            ('nan(456)', 'nan(123)', 'nan(456)'),
        ]

        for source1, source2, expected in test_triples:
            source1 = binary16(source1)
            source2 = binary16(source2)
            expected = binary16(expected)
            actual = source1.min_num_mag(source2)
            self.assertInterchangeable(
                actual,
                expected,
                'min_num_mag({0}, {1})'.format(source1, source2),
            )

    def test_max_num_mag(self):
        x = binary16('2.3')
        y = binary32('1.0')
        with self.assertRaises(ValueError):
            x.max_num_mag(y)

        test_triples = [
            # In case of equal numbers, compare signs.
            ('-0.0', '0.0', '0.0'),
            ('0.0', '-0.0', '0.0'),
            # Infinities.
            ('-inf', '-inf', '-inf'),
            ('-inf', 'inf', 'inf'),
            ('inf', '-inf', 'inf'),
            ('inf', 'inf', 'inf'),
            ('inf', '2.3', 'inf'),
            ('-inf', '2.3', '-inf'),
            ('2.3', 'inf', 'inf'),
            ('2.3', '-inf', '-inf'),
            ('-1', '1', '1'),
            ('1', '-1', '1'),
            ('1.2', '1.3', '1.3'),
            ('-1.2', '-1.3', '-1.3'),
            ('0.1', '10.0', '10.0'),
            ('-10.0', '-0.1', '-10.0'),
            # Quiet NaNs
            ('1.2', 'nan(123)', '1.2'),
            ('nan(123)', '1.2', '1.2'),
            ('nan(123)', 'nan(456)', 'nan(123)'),
            ('nan(456)', 'nan(123)', 'nan(456)'),
        ]

        for source1, source2, expected in test_triples:
            source1 = binary16(source1)
            source2 = binary16(source2)
            expected = binary16(expected)
            actual = source1.max_num_mag(source2)
            self.assertInterchangeable(
                actual,
                expected,
                'max_num_mag({0}, {1})'.format(source1, source2),
            )

    def test_scale_b(self):
        test_triples = [
            # NaNs behave as usual.
            ('nan', '0', 'nan'),
            ('-nan(123)', '54', '-nan(123)'),

            # Infinities are unchanged.
            ('inf', '0', 'inf'),
            ('inf', '1', 'inf'),
            ('inf', '-1', 'inf'),
            ('-inf', '0', '-inf'),
            ('-inf', '1', '-inf'),
            ('-inf', '-1', '-inf'),

            # So are zeros.
            ('0', '0', '0'),
            ('0', '1', '0'),
            ('0', '-1', '0'),
            ('-0', '0', '-0'),
            ('-0', '1', '-0'),
            ('-0', '-1', '-0'),

            # General case.
            ('1', '-25', '0.0'),
            ('1', '-24', '6e-8'),
            ('1', '-3', '0.125'),
            ('1', '-2', '0.25'),
            ('1', '-1', '0.5'),
            ('1', '0', '1'),
            ('1', '1', '2'),
            ('1', '2', '4'),
            ('1', '3', '8'),
            ('1', '15', '32768'),
            ('1', '16', 'inf'),

            ('-1', '-25', '-0.0'),
            ('-1', '-24', '-6e-8'),
            ('-1', '-3', '-0.125'),
            ('-1', '-2', '-0.25'),
            ('-1', '-1', '-0.5'),
            ('-1', '0', '-1'),
            ('-1', '1', '-2'),
            ('-1', '2', '-4'),
            ('-1', '3', '-8'),
            ('-1', '15', '-32768'),
            ('-1', '16', '-inf'),

            # Check rounding cases with subnormal result.
            ('0.25', '-24', '0.0'),
            ('0.5', '-24', '0.0'),
            ('0.75', '-24', '6e-8'),
            ('1', '-24', '6e-8'),
            ('1.25', '-24', '6e-8'),
            ('1.5', '-24', '1e-7'),
            ('1.75', '-24', '1e-7'),
            ('2', '-24', '1e-7'),
            ('2.25', '-24', '1e-7'),
            ('2.5', '-24', '1e-7'),
            ('2.75', '-24', '2e-7'),
        ]

        for source1, n, expected in test_triples:
            source1 = binary16(source1)
            n = int(n)
            expected = binary16(expected)
            actual = source1.scale_b(n)
            self.assertInterchangeable(
                actual,
                expected,
                'scale_b({0}, {1})'.format(source1, n),
            )

    def test_log_b(self):
        # NaNs
        for x in binary16('nan'), binary16('-snan'):
            with self.assertSignalsInvalidOperation():
                try:
                    x.log_b()
                except ValueError:
                    pass

        # Infinities
        for x in binary16('inf'), binary16('-inf'):
            with self.assertSignalsInvalidOperation():
                try:
                    x.log_b()
                except ValueError:
                    pass

        # Zeros
        for x in binary16('0'), binary16('-0'):
            with self.assertSignalsInvalidOperation():
                try:
                    x.log_b()
                except ValueError:
                    pass

        test_pairs = [
            ('0.9', '-1'),
            ('1', '0'),
            ('1.5', '0'),
            ('2.0', '1'),

            # Subnormals.
            ('6e-8', '-24'),
            ('1e-7', '-23'),
            ('2e-7', '-23'),
        ]
        for str_source1, expected in test_pairs:
            source1 = binary16(str_source1)
            expected = int(expected)
            actual = source1.log_b()
            self.assertEqual(
                actual,
                expected,
                'log_b({0}): expected {1}, got {2}'.format(
                    source1, expected, actual),
            )

            # Same test with negative values.
            source1 = binary16('-' + str_source1)
            actual = source1.log_b()
            self.assertEqual(
                actual,
                expected,
                'log_b({0}): expected {1}, got {2}'.format(
                    source1, expected, actual)
            )
