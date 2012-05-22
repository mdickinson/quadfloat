import unittest

from binary_interchange_format import BinaryInterchangeFormat

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
            (0).to_bytes(length=16, byteorder='little'),
        )
        q = Float128('3.3e-4966')
        self.assertEqual(
            q.encode(),
            (1).to_bytes(length=16, byteorder='little'),
        )
        q = Float128('-3.2e-4966')
        self.assertEqual(
            q.encode(),
            (2 ** 127).to_bytes(length=16, byteorder='little'),
        )
        q = Float128('-3.3e-4966')
        self.assertEqual(
            q.encode(),
            (1 + 2 ** 127).to_bytes(length=16, byteorder='little'),
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

    def test_negate(self):
        self.assertInterchangeable(-Float128('-2.0'), Float128('2.0'))
        self.assertInterchangeable(-Float128('2.0'), Float128('-2.0'))
        self.assertInterchangeable(-Float128('-0.0'), Float128('0.0'))
        self.assertInterchangeable(-Float128('0.0'), Float128('-0.0'))
        self.assertInterchangeable(-Float128('-inf'), Float128('inf'))
        self.assertInterchangeable(-Float128('inf'), Float128('-inf'))
        self.assertInterchangeable(-Float128('-nan'), Float128('nan'))
        self.assertInterchangeable(-Float128('nan'), Float128('-nan'))
        self.assertInterchangeable(-Float128('-snan'), Float128('snan'))
        self.assertInterchangeable(-Float128('snan'), Float128('-snan'))
        self.assertInterchangeable(-Float128('-nan(123)'), Float128('nan(123)'))
        self.assertInterchangeable(-Float128('nan(123)'), Float128('-nan(123)'))
        self.assertInterchangeable(-Float128('-snan(123)'), Float128('snan(123)'))
        self.assertInterchangeable(-Float128('snan(123)'), Float128('-snan(123)'))

    def test_abs(self):
        self.assertInterchangeable(abs(Float128('-2.0')), Float128('2.0'))
        self.assertInterchangeable(abs(Float128('2.0')), Float128('2.0'))
        self.assertInterchangeable(abs(Float128('-0.0')), Float128('0.0'))
        self.assertInterchangeable(abs(Float128('0.0')), Float128('0.0'))
        self.assertInterchangeable(abs(Float128('-inf')), Float128('inf'))
        self.assertInterchangeable(abs(Float128('inf')), Float128('inf'))
        self.assertInterchangeable(abs(Float128('-nan')), Float128('nan'))
        self.assertInterchangeable(abs(Float128('nan')), Float128('nan'))
        self.assertInterchangeable(abs(Float128('-snan')), Float128('snan'))
        self.assertInterchangeable(abs(Float128('snan')), Float128('snan'))
        self.assertInterchangeable(abs(Float128('-nan(123)')), Float128('nan(123)'))
        self.assertInterchangeable(abs(Float128('nan(123)')), Float128('nan(123)'))
        self.assertInterchangeable(abs(Float128('-snan(123)')), Float128('snan(123)'))
        self.assertInterchangeable(abs(Float128('snan(123)')), Float128('snan(123)'))


if __name__ == '__main__':
    unittest.main()

