import math
import sys
import unittest

from binary_interchange_format import BinaryInterchangeFormat

from binary_interchange_format import (
    compare_quiet_equal,
    compare_quiet_not_equal,
    compare_quiet_greater,
    compare_quiet_greater_equal,
    compare_quiet_less,
    compare_quiet_less_equal,
    compare_quiet_unordered,
    compare_quiet_not_greater,
    compare_quiet_less_unordered,
    compare_quiet_not_less,
    compare_quiet_greater_unordered,
    compare_quiet_ordered,

    compare_signaling_equal,
    compare_signaling_greater,
    compare_signaling_greater_equal,
    compare_signaling_less,
    compare_signaling_less_equal,
    compare_signaling_not_equal,
    compare_signaling_not_greater,
    compare_signaling_less_unordered,
    compare_signaling_not_less,
    compare_signaling_greater_unordered,
)

float16 = BinaryInterchangeFormat(width=16)
float32 = BinaryInterchangeFormat(width=32)
float64 = BinaryInterchangeFormat(width=64)
float128 = BinaryInterchangeFormat(width=128)
float256 = BinaryInterchangeFormat(width=256)


class TestFloat128(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two float128 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertTrue(quad1._equivalent(quad2),
                        msg = '{!r} not equivalent to {!r}'.format(quad1, quad2))

    def test_construction_no_args(self):
        q = float128()
        encoded_q = q.encode()
        self.assertIsInstance(encoded_q, bytes)
        self.assertEqual(encoded_q, b'\0'*16)

    def test_construction_from_binary_float_base(self):
        input = float128('2.3')
        q = float128(input)
        self.assertInterchangeable(q, input)

        # From smaller.
        input = float64('2.3')
        q = float128(input)
        # Value of q should be approximately 2.3; it won't
        # be exactly equal.
        absdiff = abs(q - float128('2.3'))
        self.assertLess(absdiff, 1e-15)

        # From larger.
        input = float256('2.3')
        q = float128(input)
        self.assertInterchangeable(q, float128('2.3'))

        # From an infinity.
        self.assertInterchangeable(float128(float16('inf')), float128('inf'))
        self.assertInterchangeable(float128(float256('-inf')), float128('-inf'))

        # From a NaN; check payload is clipped.
        input = float16('snan')
        self.assertInterchangeable(float128(input), float128('snan'))
        input = float16('-nan(123)')
        self.assertInterchangeable(float128(input), float128('-nan(123)'))

        input_string = 'nan({})'.format(2**230)
        input = float256(input_string)
        self.assertInterchangeable(float128(input), float128(input_string))

    def test_construction_from_int(self):
        q = float128(3)
        q = float128(-3)

        # Testing round-half-to-even.
        q = float128(5**49)
        r = float128(5**49 - 1)
        self.assertInterchangeable(q, r)

        q = float128(5**49 + 2)
        r = float128(5**49 + 3)
        self.assertInterchangeable(q, r)

        # Values near powers of two.
        for exp in range(111, 115):
            for adjust in range(-100, 100):
                n = 2 ** exp + adjust
                q = float128(n)

    def test_constructors_compatible(self):
        for n in range(-1000, 1000):
            self.assertInterchangeable(float128(n), float128(str(n)))
            self.assertInterchangeable(float128(n), float128(float(n)))

    def test_construction_from_float(self):
        q = float128(0.0)
        self.assertInterchangeable(q, float128(0))
        q = float128(1.0)
        self.assertInterchangeable(q, float128(1))
        q = float128(-13.0)
        self.assertInterchangeable(q, float128(-13))
        q = float128(float('inf'))
        self.assertInterchangeable(q, float128('inf'))
        self.assertTrue(q.is_infinite())
        q = float128(float('-inf'))
        self.assertInterchangeable(q, float128('-inf'))
        self.assertTrue(q.is_infinite())

    def test_construction_from_str(self):
        q = float128('0.0')
        self.assertInterchangeable(q, float128(0))
        q = float128('1.0')
        self.assertInterchangeable(q, float128(1))
        q = float128('-13.0')
        self.assertInterchangeable(q, float128(-13))

        # Tiny values.
        q = float128('3.2e-4966')
        self.assertEqual(
            q.encode(),
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
        )
        q = float128('3.3e-4966')
        self.assertEqual(
            q.encode(),
            b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
        )
        q = float128('-3.2e-4966')
        self.assertEqual(
            q.encode(),
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
        )
        q = float128('-3.3e-4966')
        self.assertEqual(
            q.encode(),
            b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
        )

        # Huge values.
        q = float128('1.1897e+4932')  # should be within range.
        self.assertTrue(q.is_finite())

        q = float128('1.1898e+4932')  # just overflows the range.
        self.assertTrue(q.is_infinite())

        # Infinities
        q = float128('Inf')
        self.assertTrue(q.is_infinite())
        self.assertFalse(q.is_sign_minus())

        q = float128('infinity')
        self.assertTrue(q.is_infinite())
        self.assertFalse(q.is_sign_minus())

        q = float128('-inf')
        self.assertTrue(q.is_infinite())
        self.assertTrue(q.is_sign_minus())

        q = float128('-INFINITY')
        self.assertTrue(q.is_infinite())
        self.assertTrue(q.is_sign_minus())

        # Nans with and without payloads
        for nan_string in ['nan', 'NaN', 'NAN', 'nAN', 'nan(1)', 'nan(9999)']:
            for prefix in '+', '-', '':
                q = float128(prefix + nan_string)
                self.assertTrue(q.is_nan())
                self.assertFalse(q.is_signaling())

            for prefix in '+', '-', '':
                q = float128(prefix + 's' + nan_string)
                self.assertTrue(q.is_nan())
                self.assertTrue(q.is_signaling())


        # Out-of-range payloads should just be clipped to be within range.
        q = float128('nan(123123123123123123123123123123123123)')

        with self.assertRaises(ValueError):
            float128('nan()')

        with self.assertRaises(ValueError):
            float128('+nan()')

        with self.assertRaises(ValueError):
            float128('+snan(1')

    def test_bad_constructor(self):
        with self.assertRaises(TypeError):
            float128(3.2j)

        with self.assertRaises(TypeError):
            float128([1, 2, 3])

    def test_is_canonical(self):
        self.assertTrue(float128('0.0').is_canonical())
        self.assertTrue(float128('-0.0').is_canonical())
        self.assertTrue(float128('8e-4933').is_canonical())
        self.assertTrue(float128('-8e-4933').is_canonical())
        self.assertTrue(float128('2.3').is_canonical())
        self.assertTrue(float128('-2.3').is_canonical())
        self.assertTrue(float128('Infinity').is_canonical())
        self.assertTrue(float128('-Infinity').is_canonical())
        self.assertTrue(float128('NaN').is_canonical())
        self.assertTrue(float128('-NaN').is_canonical())
        self.assertTrue(float128('sNaN').is_canonical())
        self.assertTrue(float128('-sNaN').is_canonical())

    def test_is_finite(self):
        self.assertTrue(float128('0.0').is_finite())
        self.assertTrue(float128('-0.0').is_finite())
        self.assertTrue(float128('8e-4933').is_finite())
        self.assertTrue(float128('-8e-4933').is_finite())
        self.assertTrue(float128('2.3').is_finite())
        self.assertTrue(float128('-2.3').is_finite())
        self.assertFalse(float128('Infinity').is_finite())
        self.assertFalse(float128('-Infinity').is_finite())
        self.assertFalse(float128('NaN').is_finite())
        self.assertFalse(float128('-NaN').is_finite())
        self.assertFalse(float128('sNaN').is_finite())
        self.assertFalse(float128('-sNaN').is_finite())

    def test_is_subnormal(self):
        self.assertFalse(float128('0.0').is_subnormal())
        self.assertFalse(float128('-0.0').is_subnormal())
        self.assertTrue(float128('3.3e-4932').is_subnormal())
        self.assertTrue(float128('-3.3e-4932').is_subnormal())
        self.assertFalse(float128('3.4e-4932').is_subnormal())
        self.assertFalse(float128('-3.4e-4932').is_subnormal())
        self.assertFalse(float128('2.3').is_subnormal())
        self.assertFalse(float128('-2.3').is_subnormal())
        self.assertFalse(float128('Infinity').is_subnormal())
        self.assertFalse(float128('-Infinity').is_subnormal())
        self.assertFalse(float128('NaN').is_subnormal())
        self.assertFalse(float128('-NaN').is_subnormal())
        self.assertFalse(float128('sNaN').is_subnormal())
        self.assertFalse(float128('-sNaN').is_subnormal())

    def test_is_normal(self):
        self.assertFalse(float128('0.0').is_normal())
        self.assertFalse(float128('-0.0').is_normal())
        self.assertFalse(float128('3.3e-4932').is_normal())
        self.assertFalse(float128('-3.3e-4932').is_normal())
        self.assertTrue(float128('3.4e-4932').is_normal())
        self.assertTrue(float128('-3.4e-4932').is_normal())
        self.assertTrue(float128('2.3').is_normal())
        self.assertTrue(float128('-2.3').is_normal())
        self.assertFalse(float128('Infinity').is_normal())
        self.assertFalse(float128('-Infinity').is_normal())
        self.assertFalse(float128('NaN').is_normal())
        self.assertFalse(float128('-NaN').is_normal())
        self.assertFalse(float128('sNaN').is_normal())
        self.assertFalse(float128('-sNaN').is_normal())

    def test_is_sign_minus(self):
        self.assertFalse(float128('0.0').is_sign_minus())
        self.assertTrue(float128('-0.0').is_sign_minus())
        self.assertFalse(float128('8e-4933').is_sign_minus())
        self.assertTrue(float128('-8e-4933').is_sign_minus())
        self.assertFalse(float128('2.3').is_sign_minus())
        self.assertTrue(float128('-2.3').is_sign_minus())
        self.assertFalse(float128('Infinity').is_sign_minus())
        self.assertTrue(float128('-Infinity').is_sign_minus())
        self.assertFalse(float128('NaN').is_sign_minus())
        self.assertTrue(float128('-NaN').is_sign_minus())
        self.assertFalse(float128('sNaN').is_sign_minus())
        self.assertTrue(float128('-sNaN').is_sign_minus())

    def test_is_infinite(self):
        self.assertFalse(float128('0.0').is_infinite())
        self.assertFalse(float128('-0.0').is_infinite())
        self.assertFalse(float128('8e-4933').is_infinite())
        self.assertFalse(float128('-8e-4933').is_infinite())
        self.assertFalse(float128('2.3').is_infinite())
        self.assertFalse(float128('-2.3').is_infinite())
        self.assertTrue(float128('Infinity').is_infinite())
        self.assertTrue(float128('-Infinity').is_infinite())
        self.assertFalse(float128('NaN').is_infinite())
        self.assertFalse(float128('-NaN').is_infinite())
        self.assertFalse(float128('sNaN').is_infinite())
        self.assertFalse(float128('-sNaN').is_infinite())

    def test_is_nan(self):
        self.assertFalse(float128('0.0').is_nan())
        self.assertFalse(float128('-0.0').is_nan())
        self.assertFalse(float128('8e-4933').is_nan())
        self.assertFalse(float128('-8e-4933').is_nan())
        self.assertFalse(float128('2.3').is_nan())
        self.assertFalse(float128('-2.3').is_nan())
        self.assertFalse(float128('Infinity').is_nan())
        self.assertFalse(float128('-Infinity').is_nan())
        self.assertTrue(float128('NaN').is_nan())
        self.assertTrue(float128('-NaN').is_nan())
        self.assertTrue(float128('sNaN').is_nan())
        self.assertTrue(float128('-sNaN').is_nan())

    def test_is_signaling(self):
        self.assertFalse(float128('0.0').is_signaling())
        self.assertFalse(float128('-0.0').is_signaling())
        self.assertFalse(float128('8e-4933').is_signaling())
        self.assertFalse(float128('-8e-4933').is_signaling())
        self.assertFalse(float128('2.3').is_signaling())
        self.assertFalse(float128('-2.3').is_signaling())
        self.assertFalse(float128('Infinity').is_signaling())
        self.assertFalse(float128('-Infinity').is_signaling())
        self.assertFalse(float128('NaN').is_signaling())
        self.assertFalse(float128('-NaN').is_signaling())
        self.assertTrue(float128('sNaN').is_signaling())
        self.assertTrue(float128('-sNaN').is_signaling())

    def test_is_zero(self):
        self.assertTrue(float128('0.0').is_zero())
        self.assertTrue(float128('-0.0').is_zero())
        self.assertFalse(float128('8e-4933').is_zero())
        self.assertFalse(float128('-8e-4933').is_zero())
        self.assertFalse(float128('2.3').is_zero())
        self.assertFalse(float128('-2.3').is_zero())
        self.assertFalse(float128('Infinity').is_zero())
        self.assertFalse(float128('-Infinity').is_zero())
        self.assertFalse(float128('NaN').is_zero())
        self.assertFalse(float128('-NaN').is_zero())
        self.assertFalse(float128('sNaN').is_zero())
        self.assertFalse(float128('-sNaN').is_zero())

    def test_encode(self):
        test_values = [
            (float128(0), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'),
            (float128(1), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x3f'),
            (float128(2), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x40'),
            (float128(-1), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xbf'),
            (float128(-2), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0'),
        ]
        for quad, expected in test_values:
            actual = quad.encode()
            self.assertEqual(
                actual,
                expected,
            )

    def test_encode_decode_roundtrip(self):
        test_values = [
            float128(0),
            float128(1),
            float128(-1),
            float128.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x7f'),  # inf
            float128.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff'),  # -inf
            float128.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\x7f'),  # qnan
            float128.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\xff'),  # qnan
            float128.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\x7f'),  # qnan
            float128.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\xff'),  # qnan
            float128.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x7f'),  # snan
            float128.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff'),  # snan
            float128('inf'),
            float128('-inf'),
            float128('nan'),
            float128('-nan'),
            float128('snan'),
            float128('-snan'),
        ]
        for value in test_values:
            encoded_value = value.encode()
            self.assertIsInstance(encoded_value, bytes)
            decoded_value = float128.decode(encoded_value)
            self.assertInterchangeable(value, decoded_value)

    def test_decode(self):
        # Wrong number of bytes to decode.
        with self.assertRaises(ValueError):
            float128.decode(b'\x00' * 8)

    def test_decode_encode_roundtrip(self):
        test_values = [
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            b'\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x3f',
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x40',
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xbf',
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0',
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x7f',  # inf
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff',  # -inf
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\x7f',  # qnan
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\xff',  # qnan
            b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\x7f',  # qnan
            b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\xff',  # qnan
            b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x7f',  # snan
            b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff',  # snan
        ]
        for value in test_values:
            self.assertIsInstance(value, bytes)
            decoded_value = float128.decode(value)
            encoded_value = decoded_value.encode()

            self.assertIsInstance(encoded_value, bytes)
            self.assertEqual(value, encoded_value)

    def test_repr_construct_roundtrip(self):
        test_values = [
            float128('3.2'),
            float128(3.2),
            float128(1),
            float128('-1.0'),
            float128('0.0'),
            float128('-0.0'),
            float128('3.1415926535897932384626433'),
            float128('0.1'),
            float128('0.01'),
            float128('1e1000'),
            float128('1e-1000'),
            float128(0.10000000000001e-150),
            float128(0.32e-150),
            float128(0.99999999999999e-150),
            float128(0.10000000000001e-2),
            float128(0.32e-2),
            float128(0.99999999999999e-2),
            float128(0.10000000000001e-1),
            float128(0.32e-1),
            float128(0.99999999999999e-1),
            float128(0.10000000000001),
            float128(0.32),
            float128(0.99999999999999),
            float128(1),
            float128(3.2),
            float128(9.999999999999),
            float128(10.0),
            float128(10.00000000000001),
            float128(32),
            float128(0.10000000000001e150),
            float128(0.32e150),
            float128(0.99999999999999e150),
            float128(10**200),
            float128('inf'),
            float128('-inf'),
            float128('nan'),
            float128('-nan'),
            float128('snan'),
            float128('-snan'),
            float128('nan(123)'),
            float128('-snan(999999)'),
        ]
        for value in test_values:
            repr_value = repr(value)
            reconstructed_value = eval(repr_value)
            self.assertInterchangeable(value, reconstructed_value)

            str_value = str(value)
            reconstructed_value = float128(str_value)
            self.assertInterchangeable(value, reconstructed_value)

    def test_multiplication(self):
        # First steps
        a = float128(2)
        b = float128('inf')
        self.assertInterchangeable(float128.multiplication(a, b), float128('inf'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('inf'))

        a = float128(-2)
        b = float128('inf')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-inf'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('-inf'))

        a = float128(2)
        b = float128('-inf')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-inf'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('-inf'))

        a = float128(-2)
        b = float128('-inf')
        self.assertInterchangeable(float128.multiplication(a, b), float128('inf'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('inf'))

        a = float128('0.0')
        b = float128('inf')
        self.assertTrue((float128.multiplication(a, b)).is_nan())
        self.assertTrue((float128.multiplication(b, a)).is_nan())

        a = float128('0.0')
        b = float128('0.0')
        self.assertInterchangeable(float128.multiplication(a, b), float128('0.0'))

        a = float128('-0.0')
        b = float128('0.0')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-0.0'))

        a = float128('0.0')
        b = float128('-0.0')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-0.0'))

        a = float128('-0.0')
        b = float128('-0.0')
        self.assertInterchangeable(float128.multiplication(a, b), float128('0.0'))

        a = float128('2.0')
        b = float128('0.0')
        self.assertInterchangeable(float128.multiplication(a, b), float128('0.0'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('0.0'))

        a = float128('-2.0')
        b = float128('0.0')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-0.0'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('-0.0'))

        a = float128('2.0')
        b = float128('-0.0')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-0.0'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('-0.0'))

        a = float128('-2.0')
        b = float128('-0.0')
        self.assertInterchangeable(float128.multiplication(a, b), float128('0.0'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('0.0'))

        a = float128('2.0')
        b = float128('3.0')
        self.assertInterchangeable(float128.multiplication(a, b), float128('6.0'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('6.0'))

        # signaling nans?
        a = float128('-snan(123)')
        b = float128('2.3')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('nan(456)')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('-inf')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('-2.3')
        self.assertInterchangeable(float128.multiplication(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('-nan(123)'))

        # first snan wins
        a = float128('snan(123)')
        b = float128('-snan(456)')
        self.assertInterchangeable(float128.multiplication(a, b), float128('nan(123)'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('-nan(456)'))

        # quiet nans with payload
        a = float128('2.0')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.multiplication(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('nan(789)'))

        a = float128('-2.0')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.multiplication(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('nan(789)'))

        a = float128('inf')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.multiplication(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('nan(789)'))

        a = float128('-inf')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.multiplication(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.multiplication(b, a), float128('nan(789)'))

    def test_addition(self):
        # Cases where zeros are involved.
        a = float128('0.0')
        b = float128('0.0')
        self.assertInterchangeable(float128.addition(a, b), float128('0.0'))

        a = float128('0.0')
        b = float128('-0.0')
        self.assertInterchangeable(float128.addition(a, b), float128('0.0'))
        self.assertInterchangeable(float128.addition(b, a), float128('0.0'))

        a = float128('-0.0')
        b = float128('-0.0')
        self.assertInterchangeable(float128.addition(a, b), float128('-0.0'))

        a = float128('2.0')
        b = float128('0.0')
        self.assertInterchangeable(float128.addition(a, b), float128('2.0'))
        self.assertInterchangeable(float128.addition(b, a), float128('2.0'))

        a = float128('2.0')
        b = float128('-2.0')
        self.assertInterchangeable(float128.addition(a, b), float128('0.0'))
        self.assertInterchangeable(float128.addition(b, a), float128('0.0'))

        a = float128('2.0')
        b = float128('3.0')
        self.assertInterchangeable(float128.addition(a, b), float128('5.0'))
        self.assertInterchangeable(float128.addition(b, a), float128('5.0'))

        # Infinities.
        a = float128('inf')
        b = float128('2.0')
        self.assertInterchangeable(float128.addition(a, b), float128('inf'))
        self.assertInterchangeable(float128.addition(b, a), float128('inf'))

        a = float128('inf')
        b = float128('-2.0')
        self.assertInterchangeable(float128.addition(a, b), float128('inf'))
        self.assertInterchangeable(float128.addition(b, a), float128('inf'))

        a = float128('-inf')
        b = float128('2.0')
        self.assertInterchangeable(float128.addition(a, b), float128('-inf'))
        self.assertInterchangeable(float128.addition(b, a), float128('-inf'))

        a = float128('-inf')
        b = float128('-2.0')
        self.assertInterchangeable(float128.addition(a, b), float128('-inf'))
        self.assertInterchangeable(float128.addition(b, a), float128('-inf'))

        a = float128('-inf')
        b = float128('inf')
        self.assertInterchangeable(float128.addition(a, b), float128('nan'))
        self.assertInterchangeable(float128.addition(b, a), float128('nan'))

        a = float128('inf')
        b = float128('inf')
        self.assertInterchangeable(float128.addition(a, b), float128('inf'))
        self.assertInterchangeable(float128.addition(b, a), float128('inf'))

        a = float128('-inf')
        b = float128('-inf')
        self.assertInterchangeable(float128.addition(a, b), float128('-inf'))
        self.assertInterchangeable(float128.addition(b, a), float128('-inf'))

        # signaling nans?
        a = float128('-snan(123)')
        b = float128('2.3')
        self.assertInterchangeable(float128.addition(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.addition(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('nan(456)')
        self.assertInterchangeable(float128.addition(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.addition(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('-inf')
        self.assertInterchangeable(float128.addition(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.addition(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('-2.3')
        self.assertInterchangeable(float128.addition(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.addition(b, a), float128('-nan(123)'))

        # first snan wins
        a = float128('snan(123)')
        b = float128('-snan(456)')
        self.assertInterchangeable(float128.addition(a, b), float128('nan(123)'))
        self.assertInterchangeable(float128.addition(b, a), float128('-nan(456)'))

        # quiet nans with payload
        a = float128('2.0')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.addition(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.addition(b, a), float128('nan(789)'))

        a = float128('-2.0')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.addition(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.addition(b, a), float128('nan(789)'))

        a = float128('inf')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.addition(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.addition(b, a), float128('nan(789)'))

        a = float128('-inf')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.addition(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.addition(b, a), float128('nan(789)'))

    def test_subtraction(self):
        # Cases where zeros are involved.
        a = float128('0.0')
        b = float128('0.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('0.0'))

        a = float128('0.0')
        b = float128('-0.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('0.0'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-0.0'))

        a = float128('-0.0')
        b = float128('-0.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('0.0'))

        a = float128('2.0')
        b = float128('0.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('2.0'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-2.0'))

        a = float128('2.0')
        b = float128('-2.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('4.0'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-4.0'))

        a = float128('2.0')
        b = float128('2.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('0.0'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('0.0'))

        a = float128('2.0')
        b = float128('3.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('-1.0'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('1.0'))

        # Infinities.
        a = float128('inf')
        b = float128('2.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('inf'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-inf'))

        a = float128('inf')
        b = float128('-2.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('inf'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-inf'))

        a = float128('-inf')
        b = float128('2.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('-inf'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('inf'))

        a = float128('-inf')
        b = float128('-2.0')
        self.assertInterchangeable(float128.subtraction(a, b), float128('-inf'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('inf'))

        a = float128('inf')
        b = float128('inf')
        self.assertInterchangeable(float128.subtraction(a, b), float128('nan'))

        a = float128('-inf')
        b = float128('-inf')
        self.assertInterchangeable(float128.subtraction(a, b), float128('nan'))

        a = float128('-inf')
        b = float128('inf')
        self.assertInterchangeable(float128.subtraction(a, b), float128('-inf'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('inf'))

        # signaling nans?
        a = float128('-snan(123)')
        b = float128('2.3')
        self.assertInterchangeable(float128.subtraction(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('nan(456)')
        self.assertInterchangeable(float128.subtraction(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('-inf')
        self.assertInterchangeable(float128.subtraction(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('-2.3')
        self.assertInterchangeable(float128.subtraction(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-nan(123)'))

        # first snan wins
        a = float128('snan(123)')
        b = float128('-snan(456)')
        self.assertInterchangeable(float128.subtraction(a, b), float128('nan(123)'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('-nan(456)'))

        # quiet nans with payload
        a = float128('2.0')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.subtraction(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('nan(789)'))

        a = float128('-2.0')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.subtraction(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('nan(789)'))

        a = float128('inf')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.subtraction(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('nan(789)'))

        a = float128('-inf')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.subtraction(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.subtraction(b, a), float128('nan(789)'))

    def test_division(self):
        # Finite: check all combinations of signs.
        a = float128('1.0')
        b = float128('2.0')
        self.assertInterchangeable(float128.division(a, b), float128('0.5'))
        self.assertInterchangeable(float128.division(b, a), float128('2.0'))

        a = float128('-1.0')
        b = float128('2.0')
        self.assertInterchangeable(float128.division(a, b), float128('-0.5'))
        self.assertInterchangeable(float128.division(b, a), float128('-2.0'))

        a = float128('1.0')
        b = float128('-2.0')
        self.assertInterchangeable(float128.division(a, b), float128('-0.5'))
        self.assertInterchangeable(float128.division(b, a), float128('-2.0'))

        a = float128('-1.0')
        b = float128('-2.0')
        self.assertInterchangeable(float128.division(a, b), float128('0.5'))
        self.assertInterchangeable(float128.division(b, a), float128('2.0'))

        # One or other argument zero (but not both).
        a = float128('0.0')
        b = float128('2.0')
        self.assertInterchangeable(float128.division(a, b), float128('0.0'))
        self.assertInterchangeable(float128.division(b, a), float128('inf'))

        a = float128('0.0')
        b = float128('-2.0')
        self.assertInterchangeable(float128.division(a, b), float128('-0.0'))
        self.assertInterchangeable(float128.division(b, a), float128('-inf'))

        a = float128('-0.0')
        b = float128('2.0')
        self.assertInterchangeable(float128.division(a, b), float128('-0.0'))
        self.assertInterchangeable(float128.division(b, a), float128('-inf'))

        a = float128('-0.0')
        b = float128('-2.0')
        self.assertInterchangeable(float128.division(a, b), float128('0.0'))
        self.assertInterchangeable(float128.division(b, a), float128('inf'))

        # Zero divided by zero.
        a = float128('0.0')
        b = float128('0.0')
        self.assertTrue(float128.division(a, b).is_nan())

        a = float128('-0.0')
        b = float128('0.0')
        self.assertTrue(float128.division(a, b).is_nan())

        a = float128('-0.0')
        b = float128('-0.0')
        self.assertTrue(float128.division(a, b).is_nan())

        a = float128('0.0')
        b = float128('-0.0')
        self.assertTrue(float128.division(a, b).is_nan())

        # One or other arguments is infinity.
        a = float128('inf')
        b = float128('2.3')
        self.assertInterchangeable(float128.division(a, b), float128('inf'))
        self.assertInterchangeable(float128.division(b, a), float128('0.0'))

        a = float128('-inf')
        b = float128('2.3')
        self.assertInterchangeable(float128.division(a, b), float128('-inf'))
        self.assertInterchangeable(float128.division(b, a), float128('-0.0'))

        a = float128('-inf')
        b = float128('-2.3')
        self.assertInterchangeable(float128.division(a, b), float128('inf'))
        self.assertInterchangeable(float128.division(b, a), float128('0.0'))

        a = float128('inf')
        b = float128('-2.3')
        self.assertInterchangeable(float128.division(a, b), float128('-inf'))
        self.assertInterchangeable(float128.division(b, a), float128('-0.0'))

        # Both arguments are infinity.
        a = float128('inf')
        b = float128('inf')
        self.assertTrue(float128.division(a, b).is_nan())

        a = float128('-inf')
        b = float128('inf')
        self.assertTrue(float128.division(a, b).is_nan())

        a = float128('-inf')
        b = float128('-inf')
        self.assertTrue(float128.division(a, b).is_nan())

        a = float128('inf')
        b = float128('-inf')
        self.assertTrue(float128.division(a, b).is_nan())
        
        # signaling nans?
        a = float128('-snan(123)')
        b = float128('2.3')
        self.assertInterchangeable(float128.division(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.division(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('nan(456)')
        self.assertInterchangeable(float128.division(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.division(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('-inf')
        self.assertInterchangeable(float128.division(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.division(b, a), float128('-nan(123)'))

        a = float128('-snan(123)')
        b = float128('-2.3')
        self.assertInterchangeable(float128.division(a, b), float128('-nan(123)'))
        self.assertInterchangeable(float128.division(b, a), float128('-nan(123)'))

        # first snan wins
        a = float128('snan(123)')
        b = float128('-snan(456)')
        self.assertInterchangeable(float128.division(a, b), float128('nan(123)'))
        self.assertInterchangeable(float128.division(b, a), float128('-nan(456)'))

        # quiet nans with payload
        a = float128('2.0')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.division(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.division(b, a), float128('nan(789)'))

        a = float128('-2.0')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.division(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.division(b, a), float128('nan(789)'))

        a = float128('inf')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.division(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.division(b, a), float128('nan(789)'))

        a = float128('-inf')
        b = float128('nan(789)')
        self.assertInterchangeable(float128.division(a, b), float128('nan(789)'))
        self.assertInterchangeable(float128.division(b, a), float128('nan(789)'))

        # XXX Tests for correct rounding.
        # XXX Tests for subnormal results, underflow.

    def test_fused_multiply_add(self):
        test_values = [
            # Simple cases, finite values.
            ('5', '7', '11', '46'),
            ('5', '7', '-11', '24'),
            ('5', '-7', '11', '-24'),
            ('5', '-7', '-11', '-46'),
            ('-5', '7', '11', '-24'),
            ('-5', '7', '-11', '-46'),
            ('-5', '-7', '11', '46'),
            ('-5', '-7', '-11', '24'),
            # infinities
            ('inf', '3', '2', 'inf'),
            ('3', 'inf', '2', 'inf'),
            # invalid multiplication
            ('inf', '0', '2', 'nan'),
            ('0', 'inf', '2', 'nan'),
            # A 3rd nan argument wins over an invalid multiplication.
            # Note that this differs from the decimal module;  it's
            # "implementation defined" according to 7.2(c).
            ('inf', '0', '-nan(456)', '-nan(456)'),
            ('0', 'inf', 'snan(789)', 'nan(789)'),
            # Addition of two infinities
            ('inf', '2.3', 'inf', 'inf'),
            ('inf', '2.3', '-inf', 'nan'),
            ('-inf', '2.3', '-inf', '-inf'),
            ('-inf', '2.3', 'inf', 'nan'),
            # Zeros in the multiplication.
            ('0', '2.3', '5.6', '5.6'),
            ('2.3', '0', '5.6', '5.6'),
            ('2.3', '0', '0', '0'),
            ('2.3', '0', '-0', '0'),
            ('-2.3', '0', '-0', '-0'),
            ('2.3', '-0', '-0', '-0'),
            ('-2.3', '-0', '-0', '0'),
            # Infinite 3rd argument.
            ('1.2', '2.3', '-inf', '-inf'),
            ('1.2', '2.3', 'inf', 'inf'),
            # Zero 3rd argument.
            ('12', '2.5', '0.0', '30.0'),
            ('12', '2.5', '-0.0', '30.0'),
        ]
        for strs in test_values:
            a, b, c, expected = map(float128, strs)
            self.assertInterchangeable(
                float128.fused_multiply_add(a, b, c),
                expected,
            )

    def test_convert_from_int(self):
        self.assertInterchangeable(float128.convert_from_int(5), float128('5.0'))

    def test_int(self):
        nan = float128('nan')
        with self.assertRaises(ValueError):
            int(nan)

        inf = float128('inf')
        with self.assertRaises(ValueError):
            int(inf)
        ninf = float128('-inf')
        with self.assertRaises(ValueError):
            int(ninf)

        self.assertEqual(int(float128(-1.75)), -1)
        self.assertEqual(int(float128(-1.5)), -1)
        self.assertEqual(int(float128(-1.25)), -1)
        self.assertEqual(int(float128(-1.0)), -1)
        self.assertEqual(int(float128(-0.75)), 0)
        self.assertEqual(int(float128(-0.5)), 0)
        self.assertEqual(int(float128(-0.25)), 0)
        self.assertEqual(int(float128(-0.0)), 0)
        self.assertEqual(int(float128(0.0)), 0)
        self.assertEqual(int(float128(0.25)), 0)
        self.assertEqual(int(float128(0.5)), 0)
        self.assertEqual(int(float128(0.75)), 0)
        self.assertEqual(int(float128(1.0)), 1)
        self.assertEqual(int(float128(1.25)), 1)
        self.assertEqual(int(float128(1.5)), 1)
        self.assertEqual(int(float128(1.75)), 1)

    if sys.version_info.major == 2:
        def test_long(self):
            self.assertIsInstance(long(float128(-1.75)), long)
            self.assertEqual(long(float128(-1.75)), long(-1))
            self.assertIsInstance(long(float128(2**64)), long)
            self.assertEqual(long(float128(2**64)), long(2**64))

    def test_float(self):
        self.assertTrue(math.isnan(float(float128('nan'))))
        self.assertEqual(float(float128('inf')), float('inf'))
        self.assertEqual(float(float128('-inf')), float('-inf'))
        self.assertEqual(float(float128('2.0')), 2.0)
        self.assertEqual(float(float128('-2.3')), -2.3)
        self.assertEqual(float(float128('1e400')), float('inf'))
        self.assertEqual(float(float128('-1e400')), float('-inf'))
        poszero = float(float128('0.0'))
        self.assertEqual(poszero, 0.0)
        self.assertEqual(math.copysign(1.0, poszero), math.copysign(1.0, 0.0))
        negzero = float(float128('-0.0'))
        self.assertEqual(negzero, 0.0)
        self.assertEqual(math.copysign(1.0, negzero), math.copysign(1.0, -0.0))

    def test_convert_to_integer_ties_to_even(self):
        nan = float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_ties_to_even()

        inf = float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_ties_to_even()
        ninf = float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_ties_to_even()

        self.assertEqual(float128(-1.75).convert_to_integer_ties_to_even(), -2)
        self.assertEqual(float128(-1.5).convert_to_integer_ties_to_even(), -2)
        self.assertEqual(float128(-1.25).convert_to_integer_ties_to_even(), -1)
        self.assertEqual(float128(-1.0).convert_to_integer_ties_to_even(), -1)
        self.assertEqual(float128(-0.75).convert_to_integer_ties_to_even(), -1)
        self.assertEqual(float128(-0.5).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(float128(-0.25).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(float128(-0.0).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(float128(0.0).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(float128(0.25).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(float128(0.5).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(float128(0.75).convert_to_integer_ties_to_even(), 1)
        self.assertEqual(float128(1.0).convert_to_integer_ties_to_even(), 1)
        self.assertEqual(float128(1.25).convert_to_integer_ties_to_even(), 1)
        self.assertEqual(float128(1.5).convert_to_integer_ties_to_even(), 2)
        self.assertEqual(float128(1.75).convert_to_integer_ties_to_even(), 2)

    def test_convert_to_integer_toward_zero(self):
        nan = float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_toward_zero()

        inf = float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_toward_zero()
        ninf = float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_toward_zero()

        self.assertEqual(float128(-1.75).convert_to_integer_toward_zero(), -1)
        self.assertEqual(float128(-1.5).convert_to_integer_toward_zero(), -1)
        self.assertEqual(float128(-1.25).convert_to_integer_toward_zero(), -1)
        self.assertEqual(float128(-1.0).convert_to_integer_toward_zero(), -1)
        self.assertEqual(float128(-0.75).convert_to_integer_toward_zero(), 0)
        self.assertEqual(float128(-0.5).convert_to_integer_toward_zero(), 0)
        self.assertEqual(float128(-0.25).convert_to_integer_toward_zero(), 0)
        self.assertEqual(float128(-0.0).convert_to_integer_toward_zero(), 0)
        self.assertEqual(float128(0.0).convert_to_integer_toward_zero(), 0)
        self.assertEqual(float128(0.25).convert_to_integer_toward_zero(), 0)
        self.assertEqual(float128(0.5).convert_to_integer_toward_zero(), 0)
        self.assertEqual(float128(0.75).convert_to_integer_toward_zero(), 0)
        self.assertEqual(float128(1.0).convert_to_integer_toward_zero(), 1)
        self.assertEqual(float128(1.25).convert_to_integer_toward_zero(), 1)
        self.assertEqual(float128(1.5).convert_to_integer_toward_zero(), 1)
        self.assertEqual(float128(1.75).convert_to_integer_toward_zero(), 1)

    def test_convert_to_integer_toward_positive(self):
        nan = float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_toward_positive()

        inf = float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_toward_positive()
        ninf = float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_toward_positive()

        self.assertEqual(float128(-1.75).convert_to_integer_toward_positive(), -1)
        self.assertEqual(float128(-1.5).convert_to_integer_toward_positive(), -1)
        self.assertEqual(float128(-1.25).convert_to_integer_toward_positive(), -1)
        self.assertEqual(float128(-1.0).convert_to_integer_toward_positive(), -1)
        self.assertEqual(float128(-0.75).convert_to_integer_toward_positive(), 0)
        self.assertEqual(float128(-0.5).convert_to_integer_toward_positive(), 0)
        self.assertEqual(float128(-0.25).convert_to_integer_toward_positive(), 0)
        self.assertEqual(float128(-0.0).convert_to_integer_toward_positive(), 0)
        self.assertEqual(float128(0.0).convert_to_integer_toward_positive(), 0)
        self.assertEqual(float128(0.25).convert_to_integer_toward_positive(), 1)
        self.assertEqual(float128(0.5).convert_to_integer_toward_positive(), 1)
        self.assertEqual(float128(0.75).convert_to_integer_toward_positive(), 1)
        self.assertEqual(float128(1.0).convert_to_integer_toward_positive(), 1)
        self.assertEqual(float128(1.25).convert_to_integer_toward_positive(), 2)
        self.assertEqual(float128(1.5).convert_to_integer_toward_positive(), 2)
        self.assertEqual(float128(1.75).convert_to_integer_toward_positive(), 2)

    def test_convert_to_integer_toward_negative(self):
        nan = float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_toward_negative()

        inf = float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_toward_negative()
        ninf = float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_toward_negative()

        self.assertEqual(float128(-1.75).convert_to_integer_toward_negative(), -2)
        self.assertEqual(float128(-1.5).convert_to_integer_toward_negative(), -2)
        self.assertEqual(float128(-1.25).convert_to_integer_toward_negative(), -2)
        self.assertEqual(float128(-1.0).convert_to_integer_toward_negative(), -1)
        self.assertEqual(float128(-0.75).convert_to_integer_toward_negative(), -1)
        self.assertEqual(float128(-0.5).convert_to_integer_toward_negative(), -1)
        self.assertEqual(float128(-0.25).convert_to_integer_toward_negative(), -1)
        self.assertEqual(float128(-0.0).convert_to_integer_toward_negative(), 0)
        self.assertEqual(float128(0.0).convert_to_integer_toward_negative(), 0)
        self.assertEqual(float128(0.25).convert_to_integer_toward_negative(), 0)
        self.assertEqual(float128(0.5).convert_to_integer_toward_negative(), 0)
        self.assertEqual(float128(0.75).convert_to_integer_toward_negative(), 0)
        self.assertEqual(float128(1.0).convert_to_integer_toward_negative(), 1)
        self.assertEqual(float128(1.25).convert_to_integer_toward_negative(), 1)
        self.assertEqual(float128(1.5).convert_to_integer_toward_negative(), 1)
        self.assertEqual(float128(1.75).convert_to_integer_toward_negative(), 1)

    def test_convert_to_integer_ties_to_away(self):
        nan = float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_ties_to_away()

        inf = float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_ties_to_away()
        ninf = float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_ties_to_away()

        self.assertEqual(float128(-1.75).convert_to_integer_ties_to_away(), -2)
        self.assertEqual(float128(-1.5).convert_to_integer_ties_to_away(), -2)
        self.assertEqual(float128(-1.25).convert_to_integer_ties_to_away(), -1)
        self.assertEqual(float128(-1.0).convert_to_integer_ties_to_away(), -1)
        self.assertEqual(float128(-0.75).convert_to_integer_ties_to_away(), -1)
        self.assertEqual(float128(-0.5).convert_to_integer_ties_to_away(), -1)
        self.assertEqual(float128(-0.25).convert_to_integer_ties_to_away(), 0)
        self.assertEqual(float128(-0.0).convert_to_integer_ties_to_away(), 0)
        self.assertEqual(float128(0.0).convert_to_integer_ties_to_away(), 0)
        self.assertEqual(float128(0.25).convert_to_integer_ties_to_away(), 0)
        self.assertEqual(float128(0.5).convert_to_integer_ties_to_away(), 1)
        self.assertEqual(float128(0.75).convert_to_integer_ties_to_away(), 1)
        self.assertEqual(float128(1.0).convert_to_integer_ties_to_away(), 1)
        self.assertEqual(float128(1.25).convert_to_integer_ties_to_away(), 1)
        self.assertEqual(float128(1.5).convert_to_integer_ties_to_away(), 2)
        self.assertEqual(float128(1.75).convert_to_integer_ties_to_away(), 2)

    def test_copy(self):
        self.assertInterchangeable(float128('-2.0').copy(), float128('-2.0'))
        self.assertInterchangeable(float128('2.0').copy(), float128('2.0'))
        self.assertInterchangeable(float128('-0.0').copy(), float128('-0.0'))
        self.assertInterchangeable(float128('0.0').copy(), float128('0.0'))
        self.assertInterchangeable(float128('-inf').copy(), float128('-inf'))
        self.assertInterchangeable(float128('inf').copy(), float128('inf'))
        self.assertInterchangeable(float128('-nan').copy(), float128('-nan'))
        self.assertInterchangeable(float128('nan').copy(), float128('nan'))
        self.assertInterchangeable(float128('-snan').copy(), float128('-snan'))
        self.assertInterchangeable(float128('snan').copy(), float128('snan'))
        self.assertInterchangeable(float128('-nan(123)').copy(), float128('-nan(123)'))
        self.assertInterchangeable(float128('nan(123)').copy(), float128('nan(123)'))
        self.assertInterchangeable(float128('-snan(123)').copy(), float128('-snan(123)'))
        self.assertInterchangeable(float128('snan(123)').copy(), float128('snan(123)'))

    def test_negate(self):
        self.assertInterchangeable(float128('-2.0').negate(), float128('2.0'))
        self.assertInterchangeable(float128('2.0').negate(), float128('-2.0'))
        self.assertInterchangeable(float128('-0.0').negate(), float128('0.0'))
        self.assertInterchangeable(float128('0.0').negate(), float128('-0.0'))
        self.assertInterchangeable(float128('-inf').negate(), float128('inf'))
        self.assertInterchangeable(float128('inf').negate(), float128('-inf'))
        self.assertInterchangeable(float128('-nan').negate(), float128('nan'))
        self.assertInterchangeable(float128('nan').negate(), float128('-nan'))
        self.assertInterchangeable(float128('-snan').negate(), float128('snan'))
        self.assertInterchangeable(float128('snan').negate(), float128('-snan'))
        self.assertInterchangeable(float128('-nan(123)').negate(), float128('nan(123)'))
        self.assertInterchangeable(float128('nan(123)').negate(), float128('-nan(123)'))
        self.assertInterchangeable(float128('-snan(123)').negate(), float128('snan(123)'))
        self.assertInterchangeable(float128('snan(123)').negate(), float128('-snan(123)'))

    def test_abs(self):
        self.assertInterchangeable(float128('-2.0').abs(), float128('2.0'))
        self.assertInterchangeable(float128('2.0').abs(), float128('2.0'))
        self.assertInterchangeable(float128('-0.0').abs(), float128('0.0'))
        self.assertInterchangeable(float128('0.0').abs(), float128('0.0'))
        self.assertInterchangeable(float128('-inf').abs(), float128('inf'))
        self.assertInterchangeable(float128('inf').abs(), float128('inf'))
        self.assertInterchangeable(float128('-nan').abs(), float128('nan'))
        self.assertInterchangeable(float128('nan').abs(), float128('nan'))
        self.assertInterchangeable(float128('-snan').abs(), float128('snan'))
        self.assertInterchangeable(float128('snan').abs(), float128('snan'))
        self.assertInterchangeable(float128('-nan(123)').abs(), float128('nan(123)'))
        self.assertInterchangeable(float128('nan(123)').abs(), float128('nan(123)'))
        self.assertInterchangeable(float128('-snan(123)').abs(), float128('snan(123)'))
        self.assertInterchangeable(float128('snan(123)').abs(), float128('snan(123)'))

    def test_copy_sign(self):
        self.assertInterchangeable(float128('-2.0').copy_sign(float128('1.0')), float128('2.0'))
        self.assertInterchangeable(float128('2.0').copy_sign(float128('1.0')), float128('2.0'))
        self.assertInterchangeable(float128('-0.0').copy_sign(float128('1.0')), float128('0.0'))
        self.assertInterchangeable(float128('0.0').copy_sign(float128('1.0')), float128('0.0'))
        self.assertInterchangeable(float128('-inf').copy_sign(float128('1.0')), float128('inf'))
        self.assertInterchangeable(float128('inf').copy_sign(float128('1.0')), float128('inf'))
        self.assertInterchangeable(float128('-nan').copy_sign(float128('1.0')), float128('nan'))
        self.assertInterchangeable(float128('nan').copy_sign(float128('1.0')), float128('nan'))
        self.assertInterchangeable(float128('-snan').copy_sign(float128('1.0')), float128('snan'))
        self.assertInterchangeable(float128('snan').copy_sign(float128('1.0')), float128('snan'))
        self.assertInterchangeable(float128('-nan(123)').copy_sign(float128('1.0')), float128('nan(123)'))
        self.assertInterchangeable(float128('nan(123)').copy_sign(float128('1.0')), float128('nan(123)'))
        self.assertInterchangeable(float128('-snan(123)').copy_sign(float128('1.0')), float128('snan(123)'))
        self.assertInterchangeable(float128('snan(123)').copy_sign(float128('1.0')), float128('snan(123)'))

        self.assertInterchangeable(float128('-2.0').copy_sign(float128('-1.0')), float128('-2.0'))
        self.assertInterchangeable(float128('2.0').copy_sign(float128('-1.0')), float128('-2.0'))
        self.assertInterchangeable(float128('-0.0').copy_sign(float128('-1.0')), float128('-0.0'))
        self.assertInterchangeable(float128('0.0').copy_sign(float128('-1.0')), float128('-0.0'))
        self.assertInterchangeable(float128('-inf').copy_sign(float128('-1.0')), float128('-inf'))
        self.assertInterchangeable(float128('inf').copy_sign(float128('-1.0')), float128('-inf'))
        self.assertInterchangeable(float128('-nan').copy_sign(float128('-1.0')), float128('-nan'))
        self.assertInterchangeable(float128('nan').copy_sign(float128('-1.0')), float128('-nan'))
        self.assertInterchangeable(float128('-snan').copy_sign(float128('-1.0')), float128('-snan'))
        self.assertInterchangeable(float128('snan').copy_sign(float128('-1.0')), float128('-snan'))
        self.assertInterchangeable(float128('-nan(123)').copy_sign(float128('-1.0')), float128('-nan(123)'))
        self.assertInterchangeable(float128('nan(123)').copy_sign(float128('-1.0')), float128('-nan(123)'))
        self.assertInterchangeable(float128('-snan(123)').copy_sign(float128('-1.0')), float128('-snan(123)'))
        self.assertInterchangeable(float128('snan(123)').copy_sign(float128('-1.0')), float128('-snan(123)'))

    def test_short_float_repr(self):
        x = float128('1.23456')
        self.assertEqual(str(x), '1.23456')

    # XXX Move comparison tests to a different test module?

    def _comparison_test_pairs(self):
        def pairs(seq):
            """
            Generate all pairs (x, y) from the given sequence where
            x appears before y in the sequence.

            """
            for i, upper in enumerate(seq):
                for lower in seq[:i]:
                    yield lower, upper

        EQ, LT, GT, UN = 'EQ', 'LT', 'GT', 'UN'

        zeros = [float128('0.0'), float128('-0.0')]
        positives = [
            float128('2.2'),
            float128('2.3'),
            float128('10.0'),
            float128('Infinity'),
        ]
        negatives = [x.negate() for x in positives[::-1]]
        nans = [float128('nan(123)'), float128('-nan')]

        # Non-nans are equal to themselves; all zeros are equal to all other
        # zeros.
        test_values = []
        for x in negatives + zeros + positives:
            test_values.append((x, x, EQ))
        for x, y in pairs(zeros):
            test_values.append((x, y, EQ))
            test_values.append((y, x, EQ))

        # Comparable but nonequal.
        for x, y in pairs(negatives + positives):
            test_values.append((x, y, LT))
            test_values.append((y, x, GT))
        for x in negatives:
            for y in zeros:
                test_values.append((x, y, LT))
                test_values.append((y, x, GT))
        for x in zeros:
            for y in positives:
                test_values.append((x, y, LT))
                test_values.append((y, x, GT))

        # Not comparable.
        for x in negatives + zeros + positives:
            for y in nans:
                test_values.append((x, y, UN))
                test_values.append((y, x, UN))
        for x in nans:
            test_values.append((x, x, UN))
        for x, y in pairs(nans):
            test_values.append((x, y, UN))
            test_values.append((y, x, UN))

        # Some mixed precision cases.
        all_the_same = [float16('1.25'), float32('1.25'), float64('1.25')]
        for x in all_the_same:
            for y in all_the_same:
                test_values.append((x, y, EQ))

        all_the_same = [float16('inf'), float32('inf'), float64('inf')]
        for x in all_the_same:
            for y in all_the_same:
                test_values.append((x, y, EQ))

        for x, y, relation in test_values:
            yield x, y, relation

    def _signaling_nan_pairs(self):
        snan = float128('snan')
        finite = float128('2.3')
        yield snan, finite

    # Question: how should a quiet comparison involving a signaling NaN behave?
    # The standard doesn't provide much guidance here.  Here's what we know:
    #
    #  - Comparisons are defined in a subsection of section 5.6
    #    (Signaling-computational operations).
    #
    #  - 6.2: "Signaling NaNs ... under default exception handling, signal the
    #    invalid operation exception for every ... signaling-computational
    #    operation ..."
    #
    #  - 7.2: Simply lists the operations that produce invalid operation; the
    #    default result for operations that don't produce a result in
    #    floating-point format is not given by the standard.

    def _check_quiet_compare_function(self, testfn, true_relations):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                testfn(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = testfn(x, y)
            expected = relation in true_relations
            self.assertEqual(
                actual,
                expected,
                msg="{}({}, {}):  got {!r}, expected {}.".format(
                    testfn.__name__,
                    x,
                    y,
                    actual,
                    expected,
                )
            )

    def test_quiet_comparisons(self):
        functions = [
            (compare_quiet_equal, {'EQ'}),
            (compare_quiet_not_equal, {'LT', 'GT', 'UN'}),
            (compare_quiet_greater, {'GT'}),
            (compare_quiet_greater_equal, {'GT', 'EQ'}),
            (compare_quiet_less, {'LT'}),
            (compare_quiet_unordered, {'UN'}),
            (compare_quiet_less_equal, {'LT', 'EQ'}),
            (compare_quiet_not_greater, {'LT', 'EQ', 'UN'}),
            (compare_quiet_less_unordered, {'LT', 'UN'}),
            (compare_quiet_not_less, {'GT', 'EQ', 'UN'}),
            (compare_quiet_greater_unordered, {'GT', 'UN'}),
            (compare_quiet_ordered, {'LT', 'GT', 'EQ'}),
        ]
        for function, true_relations in functions:
            self._check_quiet_compare_function(function, true_relations)

    def _check_signaling_compare_function(self, testfn, true_relations):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                testfn(x, y)

        for x, y, relation in self._comparison_test_pairs():
            if relation == 'UN':
                with self.assertRaises(ValueError):
                    testfn(x, y)
            else:
                actual = testfn(x, y)
                expected = relation in true_relations
                self.assertEqual(
                    actual,
                    expected,
                    msg="{}({}, {}):  got {!r}, expected {}.".format(
                        testfn.__name__,
                        x,
                        y,
                        actual,
                        expected,
                    )
                )

    def test_signaling_comparisons(self):
        functions = [
            (compare_signaling_equal, {'EQ'}),
            (compare_signaling_greater, {'GT'}),
            (compare_signaling_greater_equal, {'GT', 'EQ'}),
            (compare_signaling_less, {'LT'}),
            (compare_signaling_less_equal, {'LT', 'EQ'}),
            (compare_signaling_not_equal, {'LT', 'GT', 'UN'}),
            (compare_signaling_not_greater, {'LT', 'EQ', 'UN'}),
            (compare_signaling_less_unordered, {'LT', 'UN'}),
            (compare_signaling_not_less, {'GT', 'EQ', 'UN'}),
            (compare_signaling_greater_unordered, {'GT', 'UN'}),
        ]
        for function, true_relations in functions:
            self._check_signaling_compare_function(function, true_relations)


if __name__ == '__main__':
    unittest.main()
