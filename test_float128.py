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
)

Float16 = BinaryInterchangeFormat(width=16)
Float32 = BinaryInterchangeFormat(width=32)
Float64 = BinaryInterchangeFormat(width=64)
Float128 = BinaryInterchangeFormat(width=128)


class TestFloat128(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two Float128 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertTrue(quad1._equivalent(quad2),
                        msg = '{!r} not equivalent to {!r}'.format(quad1, quad2))

    def test_construction_no_args(self):
        q = Float128()
        encoded_q = q.encode()
        self.assertIsInstance(encoded_q, bytes)
        self.assertEqual(encoded_q, b'\0'*16)

    def test_construction_from_int(self):
        q = Float128(3)
        q = Float128(-3)

        # Testing round-half-to-even.
        q = Float128(5**49)
        r = Float128(5**49 - 1)
        self.assertInterchangeable(q, r)

        q = Float128(5**49 + 2)
        r = Float128(5**49 + 3)
        self.assertInterchangeable(q, r)

        # Values near powers of two.
        for exp in range(111, 115):
            for adjust in range(-100, 100):
                n = 2 ** exp + adjust
                q = Float128(n)

    def test_constructors_compatible(self):
        for n in range(-1000, 1000):
            self.assertInterchangeable(Float128(n), Float128(str(n)))
            self.assertInterchangeable(Float128(n), Float128(float(n)))

    def test_construction_from_float(self):
        q = Float128(0.0)
        self.assertInterchangeable(q, Float128(0))
        q = Float128(1.0)
        self.assertInterchangeable(q, Float128(1))
        q = Float128(-13.0)
        self.assertInterchangeable(q, Float128(-13))

    def test_construction_from_str(self):
        q = Float128('0.0')
        self.assertInterchangeable(q, Float128(0))
        q = Float128('1.0')
        self.assertInterchangeable(q, Float128(1))
        q = Float128('-13.0')
        self.assertInterchangeable(q, Float128(-13))

        # Tiny values.
        q = Float128('3.2e-4966')
        self.assertEqual(
            q.encode(),
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
        )
        q = Float128('3.3e-4966')
        self.assertEqual(
            q.encode(),
            b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
        )
        q = Float128('-3.2e-4966')
        self.assertEqual(
            q.encode(),
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
        )
        q = Float128('-3.3e-4966')
        self.assertEqual(
            q.encode(),
            b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80',
        )

        # Huge values.
        q = Float128('1.1897e+4932')  # should be within range.
        self.assertTrue(q.is_finite())

        q = Float128('1.1898e+4932')  # just overflows the range.
        self.assertTrue(q.is_infinite())

        # Infinities
        q = Float128('Inf')
        self.assertTrue(q.is_infinite())
        self.assertFalse(q.is_sign_minus())

        q = Float128('infinity')
        self.assertTrue(q.is_infinite())
        self.assertFalse(q.is_sign_minus())

        q = Float128('-inf')
        self.assertTrue(q.is_infinite())
        self.assertTrue(q.is_sign_minus())

        q = Float128('-INFINITY')
        self.assertTrue(q.is_infinite())
        self.assertTrue(q.is_sign_minus())

        # Nans with and without payloads
        for nan_string in ['nan', 'NaN', 'NAN', 'nAN', 'nan(1)', 'nan(9999)']:
            for prefix in '+', '-', '':
                q = Float128(prefix + nan_string)
                self.assertTrue(q.is_nan())
                self.assertFalse(q.is_signaling())

            for prefix in '+', '-', '':
                q = Float128(prefix + 's' + nan_string)
                self.assertTrue(q.is_nan())
                self.assertTrue(q.is_signaling())


        # Out-of-range payloads should just be clipped to be within range.
        q = Float128('nan(123123123123123123123123123123123123)')

        with self.assertRaises(ValueError):
            Float128('nan()')

        with self.assertRaises(ValueError):
            Float128('+nan()')

        with self.assertRaises(ValueError):
            Float128('+snan(1')

    def test_is_canonical(self):
        self.assertTrue(Float128('0.0').is_canonical())
        self.assertTrue(Float128('-0.0').is_canonical())
        self.assertTrue(Float128('8e-4933').is_canonical())
        self.assertTrue(Float128('-8e-4933').is_canonical())
        self.assertTrue(Float128('2.3').is_canonical())
        self.assertTrue(Float128('-2.3').is_canonical())
        self.assertTrue(Float128('Infinity').is_canonical())
        self.assertTrue(Float128('-Infinity').is_canonical())
        self.assertTrue(Float128('NaN').is_canonical())
        self.assertTrue(Float128('-NaN').is_canonical())
        self.assertTrue(Float128('sNaN').is_canonical())
        self.assertTrue(Float128('-sNaN').is_canonical())

    def test_is_finite(self):
        self.assertTrue(Float128('0.0').is_finite())
        self.assertTrue(Float128('-0.0').is_finite())
        self.assertTrue(Float128('8e-4933').is_finite())
        self.assertTrue(Float128('-8e-4933').is_finite())
        self.assertTrue(Float128('2.3').is_finite())
        self.assertTrue(Float128('-2.3').is_finite())
        self.assertFalse(Float128('Infinity').is_finite())
        self.assertFalse(Float128('-Infinity').is_finite())
        self.assertFalse(Float128('NaN').is_finite())
        self.assertFalse(Float128('-NaN').is_finite())
        self.assertFalse(Float128('sNaN').is_finite())
        self.assertFalse(Float128('-sNaN').is_finite())

    def test_is_subnormal(self):
        self.assertFalse(Float128('0.0').is_subnormal())
        self.assertFalse(Float128('-0.0').is_subnormal())
        self.assertTrue(Float128('3.3e-4932').is_subnormal())
        self.assertTrue(Float128('-3.3e-4932').is_subnormal())
        self.assertFalse(Float128('3.4e-4932').is_subnormal())
        self.assertFalse(Float128('-3.4e-4932').is_subnormal())
        self.assertFalse(Float128('2.3').is_subnormal())
        self.assertFalse(Float128('-2.3').is_subnormal())
        self.assertFalse(Float128('Infinity').is_subnormal())
        self.assertFalse(Float128('-Infinity').is_subnormal())
        self.assertFalse(Float128('NaN').is_subnormal())
        self.assertFalse(Float128('-NaN').is_subnormal())
        self.assertFalse(Float128('sNaN').is_subnormal())
        self.assertFalse(Float128('-sNaN').is_subnormal())

    def test_is_normal(self):
        self.assertFalse(Float128('0.0').is_normal())
        self.assertFalse(Float128('-0.0').is_normal())
        self.assertFalse(Float128('3.3e-4932').is_normal())
        self.assertFalse(Float128('-3.3e-4932').is_normal())
        self.assertTrue(Float128('3.4e-4932').is_normal())
        self.assertTrue(Float128('-3.4e-4932').is_normal())
        self.assertTrue(Float128('2.3').is_normal())
        self.assertTrue(Float128('-2.3').is_normal())
        self.assertFalse(Float128('Infinity').is_normal())
        self.assertFalse(Float128('-Infinity').is_normal())
        self.assertFalse(Float128('NaN').is_normal())
        self.assertFalse(Float128('-NaN').is_normal())
        self.assertFalse(Float128('sNaN').is_normal())
        self.assertFalse(Float128('-sNaN').is_normal())

    def test_is_sign_minus(self):
        self.assertFalse(Float128('0.0').is_sign_minus())
        self.assertTrue(Float128('-0.0').is_sign_minus())
        self.assertFalse(Float128('8e-4933').is_sign_minus())
        self.assertTrue(Float128('-8e-4933').is_sign_minus())
        self.assertFalse(Float128('2.3').is_sign_minus())
        self.assertTrue(Float128('-2.3').is_sign_minus())
        self.assertFalse(Float128('Infinity').is_sign_minus())
        self.assertTrue(Float128('-Infinity').is_sign_minus())
        self.assertFalse(Float128('NaN').is_sign_minus())
        self.assertTrue(Float128('-NaN').is_sign_minus())
        self.assertFalse(Float128('sNaN').is_sign_minus())
        self.assertTrue(Float128('-sNaN').is_sign_minus())

    def test_is_infinite(self):
        self.assertFalse(Float128('0.0').is_infinite())
        self.assertFalse(Float128('-0.0').is_infinite())
        self.assertFalse(Float128('8e-4933').is_infinite())
        self.assertFalse(Float128('-8e-4933').is_infinite())
        self.assertFalse(Float128('2.3').is_infinite())
        self.assertFalse(Float128('-2.3').is_infinite())
        self.assertTrue(Float128('Infinity').is_infinite())
        self.assertTrue(Float128('-Infinity').is_infinite())
        self.assertFalse(Float128('NaN').is_infinite())
        self.assertFalse(Float128('-NaN').is_infinite())
        self.assertFalse(Float128('sNaN').is_infinite())
        self.assertFalse(Float128('-sNaN').is_infinite())

    def test_is_nan(self):
        self.assertFalse(Float128('0.0').is_nan())
        self.assertFalse(Float128('-0.0').is_nan())
        self.assertFalse(Float128('8e-4933').is_nan())
        self.assertFalse(Float128('-8e-4933').is_nan())
        self.assertFalse(Float128('2.3').is_nan())
        self.assertFalse(Float128('-2.3').is_nan())
        self.assertFalse(Float128('Infinity').is_nan())
        self.assertFalse(Float128('-Infinity').is_nan())
        self.assertTrue(Float128('NaN').is_nan())
        self.assertTrue(Float128('-NaN').is_nan())
        self.assertTrue(Float128('sNaN').is_nan())
        self.assertTrue(Float128('-sNaN').is_nan())

    def test_is_signaling(self):
        self.assertFalse(Float128('0.0').is_signaling())
        self.assertFalse(Float128('-0.0').is_signaling())
        self.assertFalse(Float128('8e-4933').is_signaling())
        self.assertFalse(Float128('-8e-4933').is_signaling())
        self.assertFalse(Float128('2.3').is_signaling())
        self.assertFalse(Float128('-2.3').is_signaling())
        self.assertFalse(Float128('Infinity').is_signaling())
        self.assertFalse(Float128('-Infinity').is_signaling())
        self.assertFalse(Float128('NaN').is_signaling())
        self.assertFalse(Float128('-NaN').is_signaling())
        self.assertTrue(Float128('sNaN').is_signaling())
        self.assertTrue(Float128('-sNaN').is_signaling())

    def test_is_zero(self):
        self.assertTrue(Float128('0.0').is_zero())
        self.assertTrue(Float128('-0.0').is_zero())
        self.assertFalse(Float128('8e-4933').is_zero())
        self.assertFalse(Float128('-8e-4933').is_zero())
        self.assertFalse(Float128('2.3').is_zero())
        self.assertFalse(Float128('-2.3').is_zero())
        self.assertFalse(Float128('Infinity').is_zero())
        self.assertFalse(Float128('-Infinity').is_zero())
        self.assertFalse(Float128('NaN').is_zero())
        self.assertFalse(Float128('-NaN').is_zero())
        self.assertFalse(Float128('sNaN').is_zero())
        self.assertFalse(Float128('-sNaN').is_zero())

    def test_encode(self):
        test_values = [
            (Float128(0), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'),
            (Float128(1), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x3f'),
            (Float128(2), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x40'),
            (Float128(-1), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xbf'),
            (Float128(-2), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0'),
        ]
        for quad, expected in test_values:
            actual = quad.encode()
            self.assertEqual(
                actual,
                expected,
            )

    def test_encode_decode_roundtrip(self):
        test_values = [
            Float128(0),
            Float128(1),
            Float128(-1),
            Float128.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x7f'),  # inf
            Float128.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff'),  # -inf
            Float128.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\x7f'),  # qnan
            Float128.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\xff'),  # qnan
            Float128.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\x7f'),  # qnan
            Float128.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\xff'),  # qnan
            Float128.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x7f'),  # snan
            Float128.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff'),  # snan
            Float128('inf'),
            Float128('-inf'),
            Float128('nan'),
            Float128('-nan'),
            Float128('snan'),
            Float128('-snan'),
        ]
        for value in test_values:
            encoded_value = value.encode()
            self.assertIsInstance(encoded_value, bytes)
            decoded_value = Float128.decode(encoded_value)
            self.assertInterchangeable(value, decoded_value)

    def test_decode(self):
        # Wrong number of bytes to decode.
        with self.assertRaises(ValueError):
            Float128.decode(b'\x00' * 8)

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
            decoded_value = Float128.decode(value)
            encoded_value = decoded_value.encode()

            self.assertIsInstance(encoded_value, bytes)
            self.assertEqual(value, encoded_value)

    def test_repr_construct_roundtrip(self):
        test_values = [
            Float128('3.2'),
            Float128(3.2),
            Float128(1),
            Float128('-1.0'),
            Float128('0.0'),
            Float128('-0.0'),
            Float128('3.1415926535897932384626433'),
            Float128('0.1'),
            Float128('0.01'),
            Float128('1e1000'),
            Float128('1e-1000'),
            Float128(0.10000000000001e-150),
            Float128(0.32e-150),
            Float128(0.99999999999999e-150),
            Float128(0.10000000000001e-2),
            Float128(0.32e-2),
            Float128(0.99999999999999e-2),
            Float128(0.10000000000001e-1),
            Float128(0.32e-1),
            Float128(0.99999999999999e-1),
            Float128(0.10000000000001),
            Float128(0.32),
            Float128(0.99999999999999),
            Float128(1),
            Float128(3.2),
            Float128(9.999999999999),
            Float128(10.0),
            Float128(10.00000000000001),
            Float128(32),
            Float128(0.10000000000001e150),
            Float128(0.32e150),
            Float128(0.99999999999999e150),
            Float128(10**200),
            Float128('inf'),
            Float128('-inf'),
            Float128('nan'),
            Float128('-nan'),
            Float128('snan'),
            Float128('-snan'),
            Float128('nan(123)'),
            Float128('-snan(999999)'),
        ]
        for value in test_values:
            repr_value = repr(value)
            reconstructed_value = eval(repr_value)
            self.assertInterchangeable(value, reconstructed_value)

            str_value = str(value)
            reconstructed_value = Float128(str_value)
            self.assertInterchangeable(value, reconstructed_value)

    def test_multiplication(self):
        # First steps
        a = Float128(2)
        b = Float128('inf')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('inf'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('inf'))

        a = Float128(-2)
        b = Float128('inf')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('-inf'))

        a = Float128(2)
        b = Float128('-inf')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('-inf'))

        a = Float128(-2)
        b = Float128('-inf')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('inf'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('inf'))

        a = Float128('0.0')
        b = Float128('inf')
        self.assertTrue((Float128.multiplication(a, b)).is_nan())
        self.assertTrue((Float128.multiplication(b, a)).is_nan())

        a = Float128('0.0')
        b = Float128('0.0')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('0.0'))

        a = Float128('-0.0')
        b = Float128('0.0')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-0.0'))

        a = Float128('0.0')
        b = Float128('-0.0')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-0.0'))

        a = Float128('-0.0')
        b = Float128('-0.0')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('0.0'))

        a = Float128('2.0')
        b = Float128('0.0')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('0.0'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('0.0'))

        a = Float128('-2.0')
        b = Float128('0.0')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-0.0'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('-0.0'))

        a = Float128('2.0')
        b = Float128('-0.0')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-0.0'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('-0.0'))

        a = Float128('-2.0')
        b = Float128('-0.0')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('0.0'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('0.0'))

        a = Float128('2.0')
        b = Float128('3.0')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('6.0'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('6.0'))

        # signaling nans?
        a = Float128('-snan(123)')
        b = Float128('2.3')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('nan(456)')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('-inf')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('-2.3')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('-nan(123)'))

        # first snan wins
        a = Float128('snan(123)')
        b = Float128('-snan(456)')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('nan(123)'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('-nan(456)'))

        # quiet nans with payload
        a = Float128('2.0')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('nan(789)'))

        a = Float128('-2.0')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('nan(789)'))

        a = Float128('inf')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('nan(789)'))

        a = Float128('-inf')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.multiplication(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.multiplication(b, a), Float128('nan(789)'))

    def test_addition(self):
        # Cases where zeros are involved.
        a = Float128('0.0')
        b = Float128('0.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('0.0'))

        a = Float128('0.0')
        b = Float128('-0.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('0.0'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('0.0'))

        a = Float128('-0.0')
        b = Float128('-0.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('-0.0'))

        a = Float128('2.0')
        b = Float128('0.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('2.0'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('2.0'))

        a = Float128('2.0')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('0.0'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('0.0'))

        a = Float128('2.0')
        b = Float128('3.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('5.0'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('5.0'))

        # Infinities.
        a = Float128('inf')
        b = Float128('2.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('inf'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('inf'))

        a = Float128('inf')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('inf'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('inf'))

        a = Float128('-inf')
        b = Float128('2.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('-inf'))

        a = Float128('-inf')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.addition(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('-inf'))

        a = Float128('-inf')
        b = Float128('inf')
        self.assertInterchangeable(Float128.addition(a, b), Float128('nan'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('nan'))

        a = Float128('inf')
        b = Float128('inf')
        self.assertInterchangeable(Float128.addition(a, b), Float128('inf'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('inf'))

        a = Float128('-inf')
        b = Float128('-inf')
        self.assertInterchangeable(Float128.addition(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('-inf'))

        # signaling nans?
        a = Float128('-snan(123)')
        b = Float128('2.3')
        self.assertInterchangeable(Float128.addition(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('nan(456)')
        self.assertInterchangeable(Float128.addition(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('-inf')
        self.assertInterchangeable(Float128.addition(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('-2.3')
        self.assertInterchangeable(Float128.addition(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('-nan(123)'))

        # first snan wins
        a = Float128('snan(123)')
        b = Float128('-snan(456)')
        self.assertInterchangeable(Float128.addition(a, b), Float128('nan(123)'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('-nan(456)'))

        # quiet nans with payload
        a = Float128('2.0')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.addition(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('nan(789)'))

        a = Float128('-2.0')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.addition(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('nan(789)'))

        a = Float128('inf')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.addition(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('nan(789)'))

        a = Float128('-inf')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.addition(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.addition(b, a), Float128('nan(789)'))

    def test_subtraction(self):
        # Cases where zeros are involved.
        a = Float128('0.0')
        b = Float128('0.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('0.0'))

        a = Float128('0.0')
        b = Float128('-0.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('0.0'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-0.0'))

        a = Float128('-0.0')
        b = Float128('-0.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('0.0'))

        a = Float128('2.0')
        b = Float128('0.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('2.0'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-2.0'))

        a = Float128('2.0')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('4.0'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-4.0'))

        a = Float128('2.0')
        b = Float128('2.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('0.0'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('0.0'))

        a = Float128('2.0')
        b = Float128('3.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('-1.0'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('1.0'))

        # Infinities.
        a = Float128('inf')
        b = Float128('2.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('inf'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-inf'))

        a = Float128('inf')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('inf'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-inf'))

        a = Float128('-inf')
        b = Float128('2.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('inf'))

        a = Float128('-inf')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('inf'))

        a = Float128('inf')
        b = Float128('inf')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('nan'))

        a = Float128('-inf')
        b = Float128('-inf')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('nan'))

        a = Float128('-inf')
        b = Float128('inf')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('inf'))

        # signaling nans?
        a = Float128('-snan(123)')
        b = Float128('2.3')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('nan(456)')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('-inf')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('-2.3')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-nan(123)'))

        # first snan wins
        a = Float128('snan(123)')
        b = Float128('-snan(456)')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('nan(123)'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('-nan(456)'))

        # quiet nans with payload
        a = Float128('2.0')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('nan(789)'))

        a = Float128('-2.0')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('nan(789)'))

        a = Float128('inf')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('nan(789)'))

        a = Float128('-inf')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.subtraction(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.subtraction(b, a), Float128('nan(789)'))

    def test_division(self):
        # Finite: check all combinations of signs.
        a = Float128('1.0')
        b = Float128('2.0')
        self.assertInterchangeable(Float128.division(a, b), Float128('0.5'))
        self.assertInterchangeable(Float128.division(b, a), Float128('2.0'))

        a = Float128('-1.0')
        b = Float128('2.0')
        self.assertInterchangeable(Float128.division(a, b), Float128('-0.5'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-2.0'))

        a = Float128('1.0')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.division(a, b), Float128('-0.5'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-2.0'))

        a = Float128('-1.0')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.division(a, b), Float128('0.5'))
        self.assertInterchangeable(Float128.division(b, a), Float128('2.0'))

        # One or other argument zero (but not both).
        a = Float128('0.0')
        b = Float128('2.0')
        self.assertInterchangeable(Float128.division(a, b), Float128('0.0'))
        self.assertInterchangeable(Float128.division(b, a), Float128('inf'))

        a = Float128('0.0')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.division(a, b), Float128('-0.0'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-inf'))

        a = Float128('-0.0')
        b = Float128('2.0')
        self.assertInterchangeable(Float128.division(a, b), Float128('-0.0'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-inf'))

        a = Float128('-0.0')
        b = Float128('-2.0')
        self.assertInterchangeable(Float128.division(a, b), Float128('0.0'))
        self.assertInterchangeable(Float128.division(b, a), Float128('inf'))

        # Zero divided by zero.
        a = Float128('0.0')
        b = Float128('0.0')
        self.assertTrue(Float128.division(a, b).is_nan())

        a = Float128('-0.0')
        b = Float128('0.0')
        self.assertTrue(Float128.division(a, b).is_nan())

        a = Float128('-0.0')
        b = Float128('-0.0')
        self.assertTrue(Float128.division(a, b).is_nan())

        a = Float128('0.0')
        b = Float128('-0.0')
        self.assertTrue(Float128.division(a, b).is_nan())

        # One or other arguments is infinity.
        a = Float128('inf')
        b = Float128('2.3')
        self.assertInterchangeable(Float128.division(a, b), Float128('inf'))
        self.assertInterchangeable(Float128.division(b, a), Float128('0.0'))

        a = Float128('-inf')
        b = Float128('2.3')
        self.assertInterchangeable(Float128.division(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-0.0'))

        a = Float128('-inf')
        b = Float128('-2.3')
        self.assertInterchangeable(Float128.division(a, b), Float128('inf'))
        self.assertInterchangeable(Float128.division(b, a), Float128('0.0'))

        a = Float128('inf')
        b = Float128('-2.3')
        self.assertInterchangeable(Float128.division(a, b), Float128('-inf'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-0.0'))

        # Both arguments are infinity.
        a = Float128('inf')
        b = Float128('inf')
        self.assertTrue(Float128.division(a, b).is_nan())

        a = Float128('-inf')
        b = Float128('inf')
        self.assertTrue(Float128.division(a, b).is_nan())

        a = Float128('-inf')
        b = Float128('-inf')
        self.assertTrue(Float128.division(a, b).is_nan())

        a = Float128('inf')
        b = Float128('-inf')
        self.assertTrue(Float128.division(a, b).is_nan())
        
        # signaling nans?
        a = Float128('-snan(123)')
        b = Float128('2.3')
        self.assertInterchangeable(Float128.division(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('nan(456)')
        self.assertInterchangeable(Float128.division(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('-inf')
        self.assertInterchangeable(Float128.division(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-nan(123)'))

        a = Float128('-snan(123)')
        b = Float128('-2.3')
        self.assertInterchangeable(Float128.division(a, b), Float128('-nan(123)'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-nan(123)'))

        # first snan wins
        a = Float128('snan(123)')
        b = Float128('-snan(456)')
        self.assertInterchangeable(Float128.division(a, b), Float128('nan(123)'))
        self.assertInterchangeable(Float128.division(b, a), Float128('-nan(456)'))

        # quiet nans with payload
        a = Float128('2.0')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.division(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.division(b, a), Float128('nan(789)'))

        a = Float128('-2.0')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.division(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.division(b, a), Float128('nan(789)'))

        a = Float128('inf')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.division(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.division(b, a), Float128('nan(789)'))

        a = Float128('-inf')
        b = Float128('nan(789)')
        self.assertInterchangeable(Float128.division(a, b), Float128('nan(789)'))
        self.assertInterchangeable(Float128.division(b, a), Float128('nan(789)'))

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
            a, b, c, expected = map(Float128, strs)
            self.assertInterchangeable(
                Float128.fused_multiply_add(a, b, c),
                expected,
            )

    def test_convert_from_int(self):
        self.assertInterchangeable(Float128.convert_from_int(5), Float128('5.0'))

    def test_int(self):
        nan = Float128('nan')
        with self.assertRaises(ValueError):
            int(nan)

        inf = Float128('inf')
        with self.assertRaises(ValueError):
            int(inf)
        ninf = Float128('-inf')
        with self.assertRaises(ValueError):
            int(ninf)

        self.assertEqual(int(Float128(-1.75)), -1)
        self.assertEqual(int(Float128(-1.5)), -1)
        self.assertEqual(int(Float128(-1.25)), -1)
        self.assertEqual(int(Float128(-1.0)), -1)
        self.assertEqual(int(Float128(-0.75)), 0)
        self.assertEqual(int(Float128(-0.5)), 0)
        self.assertEqual(int(Float128(-0.25)), 0)
        self.assertEqual(int(Float128(-0.0)), 0)
        self.assertEqual(int(Float128(0.0)), 0)
        self.assertEqual(int(Float128(0.25)), 0)
        self.assertEqual(int(Float128(0.5)), 0)
        self.assertEqual(int(Float128(0.75)), 0)
        self.assertEqual(int(Float128(1.0)), 1)
        self.assertEqual(int(Float128(1.25)), 1)
        self.assertEqual(int(Float128(1.5)), 1)
        self.assertEqual(int(Float128(1.75)), 1)

    def test_convert_to_integer_ties_to_even(self):
        nan = Float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_ties_to_even()

        inf = Float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_ties_to_even()
        ninf = Float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_ties_to_even()

        self.assertEqual(Float128(-1.75).convert_to_integer_ties_to_even(), -2)
        self.assertEqual(Float128(-1.5).convert_to_integer_ties_to_even(), -2)
        self.assertEqual(Float128(-1.25).convert_to_integer_ties_to_even(), -1)
        self.assertEqual(Float128(-1.0).convert_to_integer_ties_to_even(), -1)
        self.assertEqual(Float128(-0.75).convert_to_integer_ties_to_even(), -1)
        self.assertEqual(Float128(-0.5).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(Float128(-0.25).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(Float128(-0.0).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(Float128(0.0).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(Float128(0.25).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(Float128(0.5).convert_to_integer_ties_to_even(), 0)
        self.assertEqual(Float128(0.75).convert_to_integer_ties_to_even(), 1)
        self.assertEqual(Float128(1.0).convert_to_integer_ties_to_even(), 1)
        self.assertEqual(Float128(1.25).convert_to_integer_ties_to_even(), 1)
        self.assertEqual(Float128(1.5).convert_to_integer_ties_to_even(), 2)
        self.assertEqual(Float128(1.75).convert_to_integer_ties_to_even(), 2)

    def test_convert_to_integer_toward_zero(self):
        nan = Float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_toward_zero()

        inf = Float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_toward_zero()
        ninf = Float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_toward_zero()

        self.assertEqual(Float128(-1.75).convert_to_integer_toward_zero(), -1)
        self.assertEqual(Float128(-1.5).convert_to_integer_toward_zero(), -1)
        self.assertEqual(Float128(-1.25).convert_to_integer_toward_zero(), -1)
        self.assertEqual(Float128(-1.0).convert_to_integer_toward_zero(), -1)
        self.assertEqual(Float128(-0.75).convert_to_integer_toward_zero(), 0)
        self.assertEqual(Float128(-0.5).convert_to_integer_toward_zero(), 0)
        self.assertEqual(Float128(-0.25).convert_to_integer_toward_zero(), 0)
        self.assertEqual(Float128(-0.0).convert_to_integer_toward_zero(), 0)
        self.assertEqual(Float128(0.0).convert_to_integer_toward_zero(), 0)
        self.assertEqual(Float128(0.25).convert_to_integer_toward_zero(), 0)
        self.assertEqual(Float128(0.5).convert_to_integer_toward_zero(), 0)
        self.assertEqual(Float128(0.75).convert_to_integer_toward_zero(), 0)
        self.assertEqual(Float128(1.0).convert_to_integer_toward_zero(), 1)
        self.assertEqual(Float128(1.25).convert_to_integer_toward_zero(), 1)
        self.assertEqual(Float128(1.5).convert_to_integer_toward_zero(), 1)
        self.assertEqual(Float128(1.75).convert_to_integer_toward_zero(), 1)

    def test_convert_to_integer_toward_positive(self):
        nan = Float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_toward_positive()

        inf = Float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_toward_positive()
        ninf = Float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_toward_positive()

        self.assertEqual(Float128(-1.75).convert_to_integer_toward_positive(), -1)
        self.assertEqual(Float128(-1.5).convert_to_integer_toward_positive(), -1)
        self.assertEqual(Float128(-1.25).convert_to_integer_toward_positive(), -1)
        self.assertEqual(Float128(-1.0).convert_to_integer_toward_positive(), -1)
        self.assertEqual(Float128(-0.75).convert_to_integer_toward_positive(), 0)
        self.assertEqual(Float128(-0.5).convert_to_integer_toward_positive(), 0)
        self.assertEqual(Float128(-0.25).convert_to_integer_toward_positive(), 0)
        self.assertEqual(Float128(-0.0).convert_to_integer_toward_positive(), 0)
        self.assertEqual(Float128(0.0).convert_to_integer_toward_positive(), 0)
        self.assertEqual(Float128(0.25).convert_to_integer_toward_positive(), 1)
        self.assertEqual(Float128(0.5).convert_to_integer_toward_positive(), 1)
        self.assertEqual(Float128(0.75).convert_to_integer_toward_positive(), 1)
        self.assertEqual(Float128(1.0).convert_to_integer_toward_positive(), 1)
        self.assertEqual(Float128(1.25).convert_to_integer_toward_positive(), 2)
        self.assertEqual(Float128(1.5).convert_to_integer_toward_positive(), 2)
        self.assertEqual(Float128(1.75).convert_to_integer_toward_positive(), 2)

    def test_convert_to_integer_toward_negative(self):
        nan = Float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_toward_negative()

        inf = Float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_toward_negative()
        ninf = Float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_toward_negative()

        self.assertEqual(Float128(-1.75).convert_to_integer_toward_negative(), -2)
        self.assertEqual(Float128(-1.5).convert_to_integer_toward_negative(), -2)
        self.assertEqual(Float128(-1.25).convert_to_integer_toward_negative(), -2)
        self.assertEqual(Float128(-1.0).convert_to_integer_toward_negative(), -1)
        self.assertEqual(Float128(-0.75).convert_to_integer_toward_negative(), -1)
        self.assertEqual(Float128(-0.5).convert_to_integer_toward_negative(), -1)
        self.assertEqual(Float128(-0.25).convert_to_integer_toward_negative(), -1)
        self.assertEqual(Float128(-0.0).convert_to_integer_toward_negative(), 0)
        self.assertEqual(Float128(0.0).convert_to_integer_toward_negative(), 0)
        self.assertEqual(Float128(0.25).convert_to_integer_toward_negative(), 0)
        self.assertEqual(Float128(0.5).convert_to_integer_toward_negative(), 0)
        self.assertEqual(Float128(0.75).convert_to_integer_toward_negative(), 0)
        self.assertEqual(Float128(1.0).convert_to_integer_toward_negative(), 1)
        self.assertEqual(Float128(1.25).convert_to_integer_toward_negative(), 1)
        self.assertEqual(Float128(1.5).convert_to_integer_toward_negative(), 1)
        self.assertEqual(Float128(1.75).convert_to_integer_toward_negative(), 1)

    def test_convert_to_integer_ties_to_away(self):
        nan = Float128('nan')
        with self.assertRaises(ValueError):
            nan.convert_to_integer_ties_to_away()

        inf = Float128('inf')
        with self.assertRaises(ValueError):
            inf.convert_to_integer_ties_to_away()
        ninf = Float128('-inf')
        with self.assertRaises(ValueError):
            ninf.convert_to_integer_ties_to_away()

        self.assertEqual(Float128(-1.75).convert_to_integer_ties_to_away(), -2)
        self.assertEqual(Float128(-1.5).convert_to_integer_ties_to_away(), -2)
        self.assertEqual(Float128(-1.25).convert_to_integer_ties_to_away(), -1)
        self.assertEqual(Float128(-1.0).convert_to_integer_ties_to_away(), -1)
        self.assertEqual(Float128(-0.75).convert_to_integer_ties_to_away(), -1)
        self.assertEqual(Float128(-0.5).convert_to_integer_ties_to_away(), -1)
        self.assertEqual(Float128(-0.25).convert_to_integer_ties_to_away(), 0)
        self.assertEqual(Float128(-0.0).convert_to_integer_ties_to_away(), 0)
        self.assertEqual(Float128(0.0).convert_to_integer_ties_to_away(), 0)
        self.assertEqual(Float128(0.25).convert_to_integer_ties_to_away(), 0)
        self.assertEqual(Float128(0.5).convert_to_integer_ties_to_away(), 1)
        self.assertEqual(Float128(0.75).convert_to_integer_ties_to_away(), 1)
        self.assertEqual(Float128(1.0).convert_to_integer_ties_to_away(), 1)
        self.assertEqual(Float128(1.25).convert_to_integer_ties_to_away(), 1)
        self.assertEqual(Float128(1.5).convert_to_integer_ties_to_away(), 2)
        self.assertEqual(Float128(1.75).convert_to_integer_ties_to_away(), 2)

    def test_copy(self):
        self.assertInterchangeable(Float128('-2.0').copy(), Float128('-2.0'))
        self.assertInterchangeable(Float128('2.0').copy(), Float128('2.0'))
        self.assertInterchangeable(Float128('-0.0').copy(), Float128('-0.0'))
        self.assertInterchangeable(Float128('0.0').copy(), Float128('0.0'))
        self.assertInterchangeable(Float128('-inf').copy(), Float128('-inf'))
        self.assertInterchangeable(Float128('inf').copy(), Float128('inf'))
        self.assertInterchangeable(Float128('-nan').copy(), Float128('-nan'))
        self.assertInterchangeable(Float128('nan').copy(), Float128('nan'))
        self.assertInterchangeable(Float128('-snan').copy(), Float128('-snan'))
        self.assertInterchangeable(Float128('snan').copy(), Float128('snan'))
        self.assertInterchangeable(Float128('-nan(123)').copy(), Float128('-nan(123)'))
        self.assertInterchangeable(Float128('nan(123)').copy(), Float128('nan(123)'))
        self.assertInterchangeable(Float128('-snan(123)').copy(), Float128('-snan(123)'))
        self.assertInterchangeable(Float128('snan(123)').copy(), Float128('snan(123)'))

    def test_negate(self):
        self.assertInterchangeable(Float128('-2.0').negate(), Float128('2.0'))
        self.assertInterchangeable(Float128('2.0').negate(), Float128('-2.0'))
        self.assertInterchangeable(Float128('-0.0').negate(), Float128('0.0'))
        self.assertInterchangeable(Float128('0.0').negate(), Float128('-0.0'))
        self.assertInterchangeable(Float128('-inf').negate(), Float128('inf'))
        self.assertInterchangeable(Float128('inf').negate(), Float128('-inf'))
        self.assertInterchangeable(Float128('-nan').negate(), Float128('nan'))
        self.assertInterchangeable(Float128('nan').negate(), Float128('-nan'))
        self.assertInterchangeable(Float128('-snan').negate(), Float128('snan'))
        self.assertInterchangeable(Float128('snan').negate(), Float128('-snan'))
        self.assertInterchangeable(Float128('-nan(123)').negate(), Float128('nan(123)'))
        self.assertInterchangeable(Float128('nan(123)').negate(), Float128('-nan(123)'))
        self.assertInterchangeable(Float128('-snan(123)').negate(), Float128('snan(123)'))
        self.assertInterchangeable(Float128('snan(123)').negate(), Float128('-snan(123)'))

    def test_abs(self):
        self.assertInterchangeable(Float128('-2.0').abs(), Float128('2.0'))
        self.assertInterchangeable(Float128('2.0').abs(), Float128('2.0'))
        self.assertInterchangeable(Float128('-0.0').abs(), Float128('0.0'))
        self.assertInterchangeable(Float128('0.0').abs(), Float128('0.0'))
        self.assertInterchangeable(Float128('-inf').abs(), Float128('inf'))
        self.assertInterchangeable(Float128('inf').abs(), Float128('inf'))
        self.assertInterchangeable(Float128('-nan').abs(), Float128('nan'))
        self.assertInterchangeable(Float128('nan').abs(), Float128('nan'))
        self.assertInterchangeable(Float128('-snan').abs(), Float128('snan'))
        self.assertInterchangeable(Float128('snan').abs(), Float128('snan'))
        self.assertInterchangeable(Float128('-nan(123)').abs(), Float128('nan(123)'))
        self.assertInterchangeable(Float128('nan(123)').abs(), Float128('nan(123)'))
        self.assertInterchangeable(Float128('-snan(123)').abs(), Float128('snan(123)'))
        self.assertInterchangeable(Float128('snan(123)').abs(), Float128('snan(123)'))

    def test_copy_sign(self):
        self.assertInterchangeable(Float128('-2.0').copy_sign(Float128('1.0')), Float128('2.0'))
        self.assertInterchangeable(Float128('2.0').copy_sign(Float128('1.0')), Float128('2.0'))
        self.assertInterchangeable(Float128('-0.0').copy_sign(Float128('1.0')), Float128('0.0'))
        self.assertInterchangeable(Float128('0.0').copy_sign(Float128('1.0')), Float128('0.0'))
        self.assertInterchangeable(Float128('-inf').copy_sign(Float128('1.0')), Float128('inf'))
        self.assertInterchangeable(Float128('inf').copy_sign(Float128('1.0')), Float128('inf'))
        self.assertInterchangeable(Float128('-nan').copy_sign(Float128('1.0')), Float128('nan'))
        self.assertInterchangeable(Float128('nan').copy_sign(Float128('1.0')), Float128('nan'))
        self.assertInterchangeable(Float128('-snan').copy_sign(Float128('1.0')), Float128('snan'))
        self.assertInterchangeable(Float128('snan').copy_sign(Float128('1.0')), Float128('snan'))
        self.assertInterchangeable(Float128('-nan(123)').copy_sign(Float128('1.0')), Float128('nan(123)'))
        self.assertInterchangeable(Float128('nan(123)').copy_sign(Float128('1.0')), Float128('nan(123)'))
        self.assertInterchangeable(Float128('-snan(123)').copy_sign(Float128('1.0')), Float128('snan(123)'))
        self.assertInterchangeable(Float128('snan(123)').copy_sign(Float128('1.0')), Float128('snan(123)'))

        self.assertInterchangeable(Float128('-2.0').copy_sign(Float128('-1.0')), Float128('-2.0'))
        self.assertInterchangeable(Float128('2.0').copy_sign(Float128('-1.0')), Float128('-2.0'))
        self.assertInterchangeable(Float128('-0.0').copy_sign(Float128('-1.0')), Float128('-0.0'))
        self.assertInterchangeable(Float128('0.0').copy_sign(Float128('-1.0')), Float128('-0.0'))
        self.assertInterchangeable(Float128('-inf').copy_sign(Float128('-1.0')), Float128('-inf'))
        self.assertInterchangeable(Float128('inf').copy_sign(Float128('-1.0')), Float128('-inf'))
        self.assertInterchangeable(Float128('-nan').copy_sign(Float128('-1.0')), Float128('-nan'))
        self.assertInterchangeable(Float128('nan').copy_sign(Float128('-1.0')), Float128('-nan'))
        self.assertInterchangeable(Float128('-snan').copy_sign(Float128('-1.0')), Float128('-snan'))
        self.assertInterchangeable(Float128('snan').copy_sign(Float128('-1.0')), Float128('-snan'))
        self.assertInterchangeable(Float128('-nan(123)').copy_sign(Float128('-1.0')), Float128('-nan(123)'))
        self.assertInterchangeable(Float128('nan(123)').copy_sign(Float128('-1.0')), Float128('-nan(123)'))
        self.assertInterchangeable(Float128('-snan(123)').copy_sign(Float128('-1.0')), Float128('-snan(123)'))
        self.assertInterchangeable(Float128('snan(123)').copy_sign(Float128('-1.0')), Float128('-snan(123)'))

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

        zeros = [Float128('0.0'), Float128('-0.0')]
        positives = [
            Float128('2.2'),
            Float128('2.3'),
            Float128('10.0'),
            Float128('Infinity'),
        ]
        negatives = [x.negate() for x in positives[::-1]]
        nans = [Float128('nan(123)'), Float128('-nan')]

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
        all_the_same = [Float16('1.25'), Float32('1.25'), Float64('1.25')]
        for x in all_the_same:
            for y in all_the_same:
                test_values.append((x, y, EQ))

        all_the_same = [Float16('inf'), Float32('inf'), Float64('inf')]
        for x in all_the_same:
            for y in all_the_same:
                test_values.append((x, y, EQ))

        for x, y, relation in test_values:
            yield x, y, relation

    def _signaling_nan_pairs(self):
        snan = Float128('snan')
        finite = Float128('2.3')
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

    def test_compare_quiet_equal(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_equal(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_equal(x, y)
            expected = relation == 'EQ'
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_not_equal(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_not_equal(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_not_equal(x, y)
            expected = relation != 'EQ'
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_greater(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_greater(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_greater(x, y)
            expected = relation == 'GT'
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_greater_equal(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_greater_equal(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_greater_equal(x, y)
            expected = relation in ('GT', 'EQ')
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_less(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_less(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_less(x, y)
            expected = relation == 'LT'
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_less_equal(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_less_equal(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_less_equal(x, y)
            expected = relation in ('LT', 'EQ')
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_unordered(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_unordered(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_unordered(x, y)
            expected = relation == 'UN'
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_not_greater(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_not_greater(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_not_greater(x, y)
            expected = relation != 'GT'
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_less_unordered(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_less_unordered(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_less_unordered(x, y)
            expected = relation not in ('GT', 'EQ')
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_not_less(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_not_less(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_not_less(x, y)
            expected = relation != 'LT'
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_greater_unordered(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_greater_unordered(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_greater_unordered(x, y)
            expected = relation in ('GT', 'UN')
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_compare_quiet_ordered(self):
        for x, y in self._signaling_nan_pairs():
            with self.assertRaises(ValueError):
                compare_quiet_ordered(x, y)

        for x, y, relation in self._comparison_test_pairs():
            actual = compare_quiet_ordered(x, y)
            expected = relation != 'UN'
            self.assertEqual(
                actual,
                expected,
                msg="Failed comparison: {} {} {}".format(x, y, relation),
            )

    def test_short_float_repr(self):
        x = Float128('1.23456')
        self.assertEqual(str(x), '1.23456')


if __name__ == '__main__':
    unittest.main()
