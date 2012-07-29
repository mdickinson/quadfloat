"""
Tests for mixed-precision arithmetic.

"""

import unittest

from binary_interchange_format import BinaryInterchangeFormat


float16 = BinaryInterchangeFormat(width=16)
float32 = BinaryInterchangeFormat(width=32)
float64 = BinaryInterchangeFormat(width=64)
float128 = BinaryInterchangeFormat(width=128)


class TestMixed(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two float128 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertTrue(quad1._equivalent(quad2),
                        msg = '{!r} not equivalent to {!r}'.format(quad1, quad2))

    def test_nan_payload(self):
        # a float32 NaN has 22 bits devoted to payload; float16 has only 9.
        # XXX Should really make _payload public if we're going to test it this way.
        source1 = float32('nan')
        self.assertEqual(source1._payload, 0)

        source1 = float32('nan(0)')
        self.assertEqual(source1._payload, 0)
        
        # Payloads larger than that allowed should be clipped to be within range.
        # XXX As an implementation choice, this needs documenting.
        source1 = float32('nan({})'.format(2**22))
        self.assertEqual(source1._payload, 2**22 - 1)

        source1 = float32('snan(0)')
        self.assertEqual(source1._payload, 1)

        source1 = float32('snan({})'.format(2**22))
        self.assertEqual(source1._payload, 2**22 - 1)

        source1 = float16('snan({})'.format(2**22))
        self.assertEqual(source1._payload, 2**9 - 1)

        # Now combine two float32 instances with a float16 result; NaN should be shortened
        # appropriately.
        source1 = float32('nan(999999)')
        source2 = float32('2.0')
        for op in float16.addition, float16.subtraction, float16.multiplication, float16.division:
            result = op(source1, source2)
            self.assertEqual(result._format, float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, False)

            result = op(source2, source1)
            self.assertEqual(result._format, float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, False)

        source1 = float32('-snan(999999)')
        source2 = float32('2.0')
        for op in float16.addition, float16.subtraction, float16.multiplication, float16.division:
            result = op(source1, source2)
            self.assertEqual(result._format, float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, True)

            result = op(source2, source1)
            self.assertEqual(result._format, float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, True)

    def test___pos__(self):
        a = float16('3.5')
        self.assertInterchangeable(+a, a)

        a = float16('-2.3')
        self.assertInterchangeable(+a, a)

    def test___neg__(self):
        a = float16('3.5')
        self.assertInterchangeable(-a, float16('-3.5'))

        a = float16('-2.3')
        self.assertInterchangeable(-a, float16('2.3'))

    def test___abs__(self):
        a = float16('3.5')
        self.assertInterchangeable(abs(a), a)

        a = float16('-2.3')
        self.assertInterchangeable(abs(a), -a)

    def test___add__(self):
        # Bad types.
        a = float16('3.5')
        b = '5.7'
        with self.assertRaises(TypeError):
            a + b

        # Different _BinaryInterchangeFormat subtypes.
        a = float16('3.5')
        b = float32('1.5')
        c = a + b
        self.assertEqual(c._format, float32)
        self.assertInterchangeable(c, float32('5.0'))

        c = b + a
        self.assertEqual(c._format, float32)
        self.assertInterchangeable(c, float32('5.0'))

        # Python's float treated as interchangeable with float64.
        a = 3.5
        b = float32('1.5')

        c = a + b
        self.assertEqual(c._format, float64)
        self.assertInterchangeable(c, float64('5.0'))

        c = b + a
        self.assertEqual(c._format, float64)
        self.assertInterchangeable(c, float64('5.0'))

        a = 3.5
        b = float128('1.5')

        c = a + b
        self.assertEqual(c._format, float128)
        self.assertInterchangeable(c, float128('5.0'))

        c = b + a
        self.assertEqual(c._format, float128)
        self.assertInterchangeable(c, float128('5.0'))

        # Integers are converted to the float type before the operation.
        a = float16('21')
        b = 29
        c = a + b
        self.assertEqual(c._format, float16)
        self.assertInterchangeable(c, float16('50'))

        a = float16('21')
        b = 29
        c = b + a
        self.assertEqual(c._format, float16)
        self.assertInterchangeable(c, float16('50'))

        a = float16('21')
        b = 2409
        c = a + b
        self.assertEqual(c._format, float16)
        # Note: inexact result due to rounding both of b and of the sum.
        self.assertInterchangeable(c, float16('2428'))
        c = b + a
        self.assertEqual(c._format, float16)
        # Note: inexact result due to rounding both of b and of the sum.
        self.assertInterchangeable(c, float16('2428'))

        # Ensure orders.  Addition is *not* commutative in the face of NaNs
        # with payloads!
        a = float64('NaN(123)')
        b = float('nan')
        self.assertInterchangeable(a + b, float64('NaN(123)'))
        self.assertInterchangeable(b + a, float64('NaN'))

        # If an infinity is returned, it should have the correct format.
        a = float32('3.2')
        b = float32('inf')
        c = float64.addition(a, b)
        self.assertEqual(c._format, float64)
        self.assertInterchangeable(c, float64('inf'))
        c = float64.addition(b, a)
        self.assertEqual(c._format, float64)
        self.assertInterchangeable(c, float64('inf'))
        c = float64.addition(b, b)
        self.assertEqual(c._format, float64)
        self.assertInterchangeable(c, float64('inf'))

    def test___sub__(self):
        # Bad types.
        a = float16('3.5')
        b = '5.7'
        with self.assertRaises(TypeError):
            a - b

        # Different _BinaryInterchangeFormat subtypes.
        a = float16('3.5')
        b = float32('1.5')
        c = a - b
        self.assertEqual(c._format, float32)
        self.assertInterchangeable(c, float32('2.0'))

        c = b - a
        self.assertEqual(c._format, float32)
        self.assertInterchangeable(c, float32('-2.0'))

        # Python's float treated as interchangeable with float64.
        a = 3.5
        b = float32('1.5')

        c = a - b
        self.assertEqual(c._format, float64)
        self.assertInterchangeable(c, float64('2.0'))

        c = b - a
        self.assertEqual(c._format, float64)
        self.assertInterchangeable(c, float64('-2.0'))

    def test___mul__(self):
        # Bad types.
        a = float16('3.5')
        b = '5.7'
        with self.assertRaises(TypeError):
            a * b

        # Different _BinaryInterchangeFormat subtypes.
        a = float16('3.5')
        b = float32('1.5')
        c = a * b
        self.assertEqual(c._format, float32)
        self.assertInterchangeable(c, float32('5.25'))

        c = b * a
        self.assertEqual(c._format, float32)
        self.assertInterchangeable(c, float32('5.25'))

        # Python's float treated as interchangeable with float64.
        a = 3.5
        b = float32('1.5')

        c = a * b
        self.assertEqual(c._format, float64)
        self.assertInterchangeable(c, float64('5.25'))

        c = b * a
        self.assertEqual(c._format, float64)
        self.assertInterchangeable(c, float64('5.25'))

    def test___div__(self):
        # Bad types.
        a = float16('3.5')
        b = '5.7'
        with self.assertRaises(TypeError):
            a / b

        a = float16('35.0')
        b = float32('5.0')
        c = a / b
        self.assertEqual(c._format, float32)
        self.assertInterchangeable(c, float32('7.0'))

        c = b / a
        self.assertEqual(c._format, float32)
        self.assertInterchangeable(c, float32('0.142857142857142857142857'))

    def test_mixed_arithmetic(self):
        # Check that large integers work.
        a = float32(0)
        b = 2**64
        c = a + b
        self.assertEqual(c._format, float32)
        self.assertInterchangeable(c, float32(2**64))

    def test_equality_operation(self):
        # Comparisons between integers and floats.
        self.assertFalse(float64(3.2) == 0)
        self.assertTrue(float64(0) == 0)
        # Check that we're not simply converting the integer to a float.
        self.assertFalse(float64(2**64) == 2 ** 64 + 1)
        self.assertTrue(float64(2**64) == 2 ** 64)


if __name__ == '__main__':
    unittest.main()
