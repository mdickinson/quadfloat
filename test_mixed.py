"""
Tests for mixed-precision arithmetic.

"""

import unittest

from binary_interchange_format import BinaryInterchangeFormat


Float16 = BinaryInterchangeFormat(width=16)
Float32 = BinaryInterchangeFormat(width=32)
Float64 = BinaryInterchangeFormat(width=64)
Float128 = BinaryInterchangeFormat(width=128)


class TestMixed(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two Float128 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertTrue(quad1._equivalent(quad2),
                        msg = '{!r} not equivalent to {!r}'.format(quad1, quad2))

    def test_nan_payload(self):
        # a Float32 NaN has 22 bits devoted to payload; Float16 has only 9.
        # XXX Should really make _payload public if we're going to test it this way.
        source1 = Float32('nan')
        self.assertEqual(source1._payload, 0)

        source1 = Float32('nan(0)')
        self.assertEqual(source1._payload, 0)
        
        # Payloads larger than that allowed should be clipped to be within range.
        # XXX As an implementation choice, this needs documenting.
        source1 = Float32('nan({})'.format(2**22))
        self.assertEqual(source1._payload, 2**22 - 1)

        source1 = Float32('snan(0)')
        self.assertEqual(source1._payload, 1)

        source1 = Float32('snan({})'.format(2**22))
        self.assertEqual(source1._payload, 2**22 - 1)

        source1 = Float16('snan({})'.format(2**22))
        self.assertEqual(source1._payload, 2**9 - 1)

        # Now combine two Float32 instances with a Float16 result; NaN should be shortened
        # appropriately.
        source1 = Float32('nan(999999)')
        source2 = Float32('2.0')
        for op in Float16.addition, Float16.subtraction, Float16.multiplication, Float16.division:
            result = op(source1, source2)
            self.assertEqual(result._format, Float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, False)

            result = op(source2, source1)
            self.assertEqual(result._format, Float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, False)

        source1 = Float32('-snan(999999)')
        source2 = Float32('2.0')
        for op in Float16.addition, Float16.subtraction, Float16.multiplication, Float16.division:
            result = op(source1, source2)
            self.assertEqual(result._format, Float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, True)

            result = op(source2, source1)
            self.assertEqual(result._format, Float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, True)

    def test_addition(self):
        # Different _BinaryInterchangeFormat subtypes.
        a = Float16('3.5')
        b = Float32('1.5')
        c = a + b
        self.assertEqual(c._format, Float32)
        self.assertInterchangeable(c, Float32('5.0'))

        c = b + a
        self.assertEqual(c._format, Float32)
        self.assertInterchangeable(c, Float32('5.0'))

        # Python's float treated as interchangeable with float64.
        a = 3.5
        b = Float32('1.5')

        c = a + b
        self.assertEqual(c._format, Float64)
        self.assertInterchangeable(c, Float64('5.0'))

        c = b + a
        self.assertEqual(c._format, Float64)
        self.assertInterchangeable(c, Float64('5.0'))

        a = 3.5
        b = Float128('1.5')

        c = a + b
        self.assertEqual(c._format, Float128)
        self.assertInterchangeable(c, Float128('5.0'))

        c = b + a
        self.assertEqual(c._format, Float128)
        self.assertInterchangeable(c, Float128('5.0'))

        # Integers are converted to the float type before the operation.
        a = Float16('21')
        b = 29
        c = a + b
        self.assertEqual(c._format, Float16)
        self.assertInterchangeable(c, Float16('50'))

        a = Float16('21')
        b = 29
        c = b + a
        self.assertEqual(c._format, Float16)
        self.assertInterchangeable(c, Float16('50'))

        a = Float16('21')
        b = 2409
        c = a + b
        self.assertEqual(c._format, Float16)
        # Note: inexact result due to rounding both of b and of the sum.
        self.assertInterchangeable(c, Float16('2428'))
        c = b + a
        self.assertEqual(c._format, Float16)
        # Note: inexact result due to rounding both of b and of the sum.
        self.assertInterchangeable(c, Float16('2428'))

        # Ensure orders.  Addition is *not* commutative in the face of NaNs
        # with payloads!
        a = Float64('NaN(123)')
        b = float('nan')
        self.assertInterchangeable(a + b, Float64('NaN(123)'))
        self.assertInterchangeable(b + a, Float64('NaN'))

        # If an infinity is returned, it should have the correct format.
        a = Float32('3.2')
        b = Float32('inf')
        c = Float64.addition(a, b)
        self.assertEqual(c._format, Float64)
        self.assertInterchangeable(c, Float64('inf'))
        c = Float64.addition(b, a)
        self.assertEqual(c._format, Float64)
        self.assertInterchangeable(c, Float64('inf'))
        c = Float64.addition(b, b)
        self.assertEqual(c._format, Float64)
        self.assertInterchangeable(c, Float64('inf'))

    def test_subtraction(self):
        # Different _BinaryInterchangeFormat subtypes.
        a = Float16('3.5')
        b = Float32('1.5')
        c = a - b
        self.assertEqual(c._format, Float32)
        self.assertInterchangeable(c, Float32('2.0'))

        c = b - a
        self.assertEqual(c._format, Float32)
        self.assertInterchangeable(c, Float32('-2.0'))

        # Python's float treated as interchangeable with float64.
        a = 3.5
        b = Float32('1.5')

        c = a - b
        self.assertEqual(c._format, Float64)
        self.assertInterchangeable(c, Float64('2.0'))

        c = b - a
        self.assertEqual(c._format, Float64)
        self.assertInterchangeable(c, Float64('-2.0'))

    def test_multiplication(self):
        # Different _BinaryInterchangeFormat subtypes.
        a = Float16('3.5')
        b = Float32('1.5')
        c = a * b
        self.assertEqual(c._format, Float32)
        self.assertInterchangeable(c, Float32('5.25'))

        c = b * a
        self.assertEqual(c._format, Float32)
        self.assertInterchangeable(c, Float32('5.25'))

        # Python's float treated as interchangeable with float64.
        a = 3.5
        b = Float32('1.5')

        c = a * b
        self.assertEqual(c._format, Float64)
        self.assertInterchangeable(c, Float64('5.25'))

        c = b * a
        self.assertEqual(c._format, Float64)
        self.assertInterchangeable(c, Float64('5.25'))

    def test_division(self):
        a = Float16('35.0')
        b = Float32('5.0')
        c = a / b
        self.assertEqual(c._format, Float32)
        self.assertInterchangeable(c, Float32('7.0'))

        c = b / a
        self.assertEqual(c._format, Float32)
        self.assertInterchangeable(c, Float32('0.142857142857142857142857'))

    def test_mixed_arithmetic(self):
        # Check that large integers work.
        a = Float32(0)
        b = 2**64
        c = a + b
        self.assertEqual(c._format, Float32)
        self.assertInterchangeable(c, Float32(2**64))

    def test_equality_operation(self):
        # Comparisons between integers and floats.
        self.assertFalse(Float64(3.2) == 0)
        self.assertTrue(Float64(0) == 0)
        # Check that we're not simply converting the integer to a float.
        self.assertFalse(Float64(2**64) == 2 ** 64 + 1)
        self.assertTrue(Float64(2**64) == 2 ** 64)


if __name__ == '__main__':
    unittest.main()
