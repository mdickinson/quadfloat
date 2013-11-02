"""
Tests for conversion of a finite BinaryFloat object to a decimal triple.
These conversions form the basis for the convertToDecimalCharacter function.

"""
import unittest

from quadfloat.api import binary64, binary128
from quadfloat.binary_interchange_format import (
    _base_10_exponent,
    _fix_decimal_exponent,
)


class TestConvertToDecimal(unittest.TestCase):
    def test__fix_decimal_exponent(self):
        # An approximation to pi.
        x = binary64.convert_from_hex_character('0x1.921fb54442d18p+1')

        # Should return a triple (sign, exponent, significand), to be
        # interpreted as (-1)**sign * 10**exponent * signficand.
        self.assertEqual(
            _fix_decimal_exponent(x, places=-51),
            (-51, 3141592653589793115997963468544185161590576171875000))
        self.assertEqual(
            _fix_decimal_exponent(x, places=-20),
            (-20, 314159265358979311600))
        self.assertEqual(
            _fix_decimal_exponent(x, places=-6),
            (-6, 3141593))
        self.assertEqual(
            _fix_decimal_exponent(x, places=-3),
            (-3, 3142))
        self.assertEqual(
            _fix_decimal_exponent(x, places=-1),
            (-1, 31))
        self.assertEqual(
            _fix_decimal_exponent(x, places=0),
            (0, 3))
        self.assertEqual(
            _fix_decimal_exponent(x, places=1),
            (1, 0))

        y = binary64.convert_from_int(299792458)
        self.assertEqual(
            _fix_decimal_exponent(y, places=-6),
            (-6, 299792458000000))
        self.assertEqual(
            _fix_decimal_exponent(y, places=-3),
            (-3, 299792458000))
        self.assertEqual(
            _fix_decimal_exponent(y, places=0),
            (0, 299792458))
        self.assertEqual(
            _fix_decimal_exponent(y, places=1),
            (1, 29979246))
        self.assertEqual(
            _fix_decimal_exponent(y, places=2),
            (2, 2997925))
        self.assertEqual(
            _fix_decimal_exponent(y, places=3),
            (3, 299792))
        self.assertEqual(
            _fix_decimal_exponent(y, places=6),
            (6, 300))
        self.assertEqual(
            _fix_decimal_exponent(y, places=8),
            (8, 3))
        self.assertEqual(
            _fix_decimal_exponent(y, places=9),
            (9, 0))

    def test__base_10_exponent(self):
        for n in range(-100, 100):
            test_string = '0.99999999999999999999e{}'.format(n)
            test_value = binary128(test_string)
            self.assertEqual(_base_10_exponent(test_value), n)
        for n in range(-100, 100):
            test_string = '0.10000000000000000001e{}'.format(n)
            test_value = binary128(test_string)
            self.assertEqual(_base_10_exponent(test_value), n)
        for n in range(49):
            test_string = '1e{}'.format(n)
            test_value = binary128(test_string)
            self.assertEqual(_base_10_exponent(test_value), n + 1)
