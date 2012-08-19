import math
import unittest

from quadfloat.binary_interchange_format import BinaryInterchangeFormat


class TestBinaryIntegerchangeFormat(unittest.TestCase):
    def test_parameters(self):
        binary128 = BinaryInterchangeFormat(width=128)
        self.assertEqual(binary128.width, 128)
        self.assertEqual(binary128.precision, 113)
        self.assertEqual(binary128.emax, 16383)
        self.assertEqual(binary128.emin, -16382)

    def test_read_only_width(self):
        binary64 = BinaryInterchangeFormat(width=128)
        with self.assertRaises(AttributeError):
            binary64.width = 64
        with self.assertRaises(AttributeError):
            binary64.precision = 53

    def test_bad_width(self):
        with self.assertRaises(ValueError):
            binary_bad = BinaryInterchangeFormat(width=96)

    def test_precision_formula(self):
        binary16 = BinaryInterchangeFormat(width=16)
        self.assertEqual(binary16.precision, 11)

        binary32 = BinaryInterchangeFormat(width=32)
        self.assertEqual(binary32.precision, 24)

        binary64 = BinaryInterchangeFormat(width=64)
        self.assertEqual(binary64.precision, 53)

        binary128 = BinaryInterchangeFormat(width=128)
        self.assertEqual(binary128.precision, 113)

        binary256 = BinaryInterchangeFormat(width=256)
        self.assertEqual(binary256.precision, 237)

        for width in range(128, 100000, 32):
            format = BinaryInterchangeFormat(width=width)
            expected_precision = width - round(4.0 * math.log(width) / math.log(2)) + 13
            actual_precision = format.precision
            self.assertEqual(actual_precision, expected_precision)

    def test_equality(self):
        binary128 = BinaryInterchangeFormat(width=128)
        binary128_copy = BinaryInterchangeFormat(width=128)
        binary256 = BinaryInterchangeFormat(width=256)
        self.assertTrue(binary128 == binary128)
        self.assertTrue(hash(binary128) == hash(binary128))
        self.assertTrue(binary128 == binary128_copy)
        self.assertTrue(hash(binary128) == hash(binary128_copy))
        self.assertFalse(binary128 == binary256)
        self.assertFalse(binary128 != binary128)
        self.assertFalse(binary128 != binary128_copy)
        self.assertTrue(binary128 != binary256)


if __name__ == '__main__':
    unittest.main()

