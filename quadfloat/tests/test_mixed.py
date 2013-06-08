"""
Tests for mixed-precision arithmetic.

"""
from quadfloat import binary16, binary32, binary64, binary128
from quadfloat.tests.base_test_case import BaseTestCase


class TestMixed(BaseTestCase):
    def test_nan_payload(self):
        # a binary32 NaN has 22 bits devoted to payload; binary16 has only 9.

        # XXX Should really make _payload public if we're going to test it this
        # way.

        source1 = binary32('nan')
        self.assertEqual(source1._payload, 0)

        source1 = binary32('nan(0)')
        self.assertEqual(source1._payload, 0)

        # Payloads larger than that allowed should be clipped to be within
        # range. XXX As an implementation choice, this needs documenting.

        source1 = binary32('nan({0})'.format(2 ** 22))
        self.assertEqual(source1._payload, 2 ** 22 - 1)

        source1 = binary32('snan(0)')
        self.assertEqual(source1._payload, 1)

        source1 = binary32('snan({0})'.format(2 ** 22))
        self.assertEqual(source1._payload, 2 ** 22 - 1)

        source1 = binary16('snan({0})'.format(2 ** 22))
        self.assertEqual(source1._payload, 2 ** 9 - 1)

        # Now combine two binary32 instances with a binary16 result; NaN should
        # be shortened appropriately.
        source1 = binary32('nan(999999)')
        source2 = binary32('2.0')
        operations = [
            binary16.addition,
            binary16.subtraction,
            binary16.multiplication,
            binary16.division,
        ]
        for op in operations:
            result = op(source1, source2)
            self.assertEqual(result.format, binary16)
            self.assertEqual(result._payload, 2 ** 9 - 1)
            self.assertEqual(result._sign, False)

            result = op(source2, source1)
            self.assertEqual(result.format, binary16)
            self.assertEqual(result._payload, 2 ** 9 - 1)
            self.assertEqual(result._sign, False)

        source1 = binary32('-snan(999999)')
        source2 = binary32('2.0')
        for op in operations:
            result = op(source1, source2)
            self.assertEqual(result.format, binary16)
            self.assertEqual(result._payload, 2 ** 9 - 1)
            self.assertEqual(result._sign, True)

            result = op(source2, source1)
            self.assertEqual(result.format, binary16)
            self.assertEqual(result._payload, 2 ** 9 - 1)
            self.assertEqual(result._sign, True)

    def test___pos__(self):
        a = binary16('3.5')
        self.assertInterchangeable(+a, a)

        a = binary16('-2.3')
        self.assertInterchangeable(+a, a)

    def test___neg__(self):
        a = binary16('3.5')
        self.assertInterchangeable(-a, binary16('-3.5'))

        a = binary16('-2.3')
        self.assertInterchangeable(-a, binary16('2.3'))

    def test___abs__(self):
        a = binary16('3.5')
        self.assertInterchangeable(abs(a), a)

        a = binary16('-2.3')
        self.assertInterchangeable(abs(a), -a)

    def test___add__(self):
        # Bad types.
        a = binary16('3.5')
        b = '5.7'
        with self.assertRaises(TypeError):
            a + b

        # Different formats.
        a = binary16('3.5')
        b = binary32('1.5')
        c = a + b
        self.assertEqual(c.format, binary32)
        self.assertInterchangeable(c, binary32('5.0'))

        c = b + a
        self.assertEqual(c.format, binary32)
        self.assertInterchangeable(c, binary32('5.0'))

        # Python's float treated as interchangeable with binary64.
        a = 3.5
        b = binary32('1.5')

        c = a + b
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64('5.0'))

        c = b + a
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64('5.0'))

        a = 3.5
        b = binary128('1.5')

        c = a + b
        self.assertEqual(c.format, binary128)
        self.assertInterchangeable(c, binary128('5.0'))

        c = b + a
        self.assertEqual(c.format, binary128)
        self.assertInterchangeable(c, binary128('5.0'))

        # Integers are converted to the float type before the operation.
        a = binary16('21')
        b = 29
        c = a + b
        self.assertEqual(c.format, binary16)
        self.assertInterchangeable(c, binary16('50'))

        a = binary16('21')
        b = 29
        c = b + a
        self.assertEqual(c.format, binary16)
        self.assertInterchangeable(c, binary16('50'))

        a = binary16('21')
        b = 2409
        c = a + b
        self.assertEqual(c.format, binary16)
        # Note: inexact result due to rounding both of b and of the sum.
        self.assertInterchangeable(c, binary16('2428'))
        c = b + a
        self.assertEqual(c.format, binary16)
        # Note: inexact result due to rounding both of b and of the sum.
        self.assertInterchangeable(c, binary16('2428'))

        # Ensure orders.  Addition is *not* commutative in the face of NaNs
        # with payloads!
        a = binary64('NaN(123)')
        b = float('nan')
        self.assertInterchangeable(a + b, binary64('NaN(123)'))
        self.assertInterchangeable(b + a, binary64('NaN'))

        # If an infinity is returned, it should have the correct format.
        a = binary32('3.2')
        b = binary32('inf')
        c = binary64.addition(a, b)
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64('inf'))
        c = binary64.addition(b, a)
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64('inf'))
        c = binary64.addition(b, b)
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64('inf'))

    def test___sub__(self):
        # Bad types.
        a = binary16('3.5')
        b = '5.7'
        with self.assertRaises(TypeError):
            a - b

        # Different formats.
        a = binary16('3.5')
        b = binary32('1.5')
        c = a - b
        self.assertEqual(c.format, binary32)
        self.assertInterchangeable(c, binary32('2.0'))

        c = b - a
        self.assertEqual(c.format, binary32)
        self.assertInterchangeable(c, binary32('-2.0'))

        # Python's float treated as interchangeable with binary64.
        a = 3.5
        b = binary32('1.5')

        c = a - b
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64('2.0'))

        c = b - a
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64('-2.0'))

    def test___mul__(self):
        # Bad types.
        a = binary16('3.5')
        b = '5.7'
        with self.assertRaises(TypeError):
            a * b

        # Different formats.
        a = binary16('3.5')
        b = binary32('1.5')
        c = a * b
        self.assertEqual(c.format, binary32)
        self.assertInterchangeable(c, binary32('5.25'))

        c = b * a
        self.assertEqual(c.format, binary32)
        self.assertInterchangeable(c, binary32('5.25'))

        # Python's float treated as interchangeable with binary64.
        a = 3.5
        b = binary32('1.5')

        c = a * b
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64('5.25'))

        c = b * a
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64('5.25'))

    def test___div__(self):
        # Bad types.
        a = binary16('3.5')
        b = '5.7'
        with self.assertRaises(TypeError):
            a / b

        a = binary16('35.0')
        b = binary32('5.0')
        c = a / b
        self.assertEqual(c.format, binary32)
        self.assertInterchangeable(c, binary32('7.0'))

        c = b / a
        self.assertEqual(c.format, binary32)
        self.assertInterchangeable(c, binary32('0.142857142857142857142857'))

        # Python's float treated as interchangeable with binary64.
        a = 3.5
        b = binary32('1.5')

        c = a / b
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64(3.5 / 1.5))

        c = b / a
        self.assertEqual(c.format, binary64)
        self.assertInterchangeable(c, binary64(1.5 / 3.5))

    def test_mixed_arithmetic(self):
        # Check that large integers work.
        a = binary32(0)
        b = 2 ** 64
        c = a + b
        self.assertEqual(c.format, binary32)
        self.assertInterchangeable(c, binary32(2 ** 64))

    def test_equality_operation(self):
        # Comparisons between integers and floats.
        self.assertFalse(binary64(3.2) == 0)
        self.assertTrue(binary64(0) == 0)
        # Check that we're not simply converting the integer to a float.
        self.assertFalse(binary64(2 ** 64) == 2 ** 64 + 1)
        self.assertTrue(binary64(2 ** 64) == 2 ** 64)

        with self.assertRaises(TypeError):
            binary64(3.2) == '3.2'
