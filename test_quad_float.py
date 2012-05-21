import unittest

from quad_float import BinaryInterchangeFormat

QuadFloat = BinaryInterchangeFormat(width=128)


class TestQuadFloat(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two QuadFloat instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertTrue(quad1._equivalent(quad2),
                        msg = '{!r} not equivalent to {!r}'.format(quad1, quad2))

    def test_construction_no_args(self):
        q = QuadFloat()
        encoded_q = q.encode()
        self.assertIsInstance(encoded_q, bytes)
        self.assertEqual(encoded_q, b'\0'*16)

    def test_construction_from_int(self):
        q = QuadFloat(3)
        q = QuadFloat(-3)

        # Testing round-half-to-even.
        q = QuadFloat(5**49)
        r = QuadFloat(5**49 - 1)
        self.assertInterchangeable(q, r)

        q = QuadFloat(5**49 + 2)
        r = QuadFloat(5**49 + 3)
        self.assertInterchangeable(q, r)

        # Values near powers of two.
        for exp in range(111, 115):
            for adjust in range(-100, 100):
                n = 2 ** exp + adjust
                q = QuadFloat(n)

    def test_constructors_compatible(self):
        for n in range(-1000, 1000):
            self.assertInterchangeable(QuadFloat(n), QuadFloat(str(n)))
            self.assertInterchangeable(QuadFloat(n), QuadFloat(float(n)))

    def test_construction_from_float(self):
        q = QuadFloat(0.0)
        self.assertInterchangeable(q, QuadFloat(0))
        q = QuadFloat(1.0)
        self.assertInterchangeable(q, QuadFloat(1))
        q = QuadFloat(-13.0)
        self.assertInterchangeable(q, QuadFloat(-13))

    def test_construction_from_str(self):
        q = QuadFloat('0.0')
        self.assertInterchangeable(q, QuadFloat(0))
        q = QuadFloat('1.0')
        self.assertInterchangeable(q, QuadFloat(1))
        q = QuadFloat('-13.0')
        self.assertInterchangeable(q, QuadFloat(-13))

        # Tiny values.
        q = QuadFloat('3.2e-4966')
        self.assertEqual(
            q.encode(),
            (0).to_bytes(length=16, byteorder='little'),
        )
        q = QuadFloat('3.3e-4966')
        self.assertEqual(
            q.encode(),
            (1).to_bytes(length=16, byteorder='little'),
        )
        q = QuadFloat('-3.2e-4966')
        self.assertEqual(
            q.encode(),
            (2 ** 127).to_bytes(length=16, byteorder='little'),
        )
        q = QuadFloat('-3.3e-4966')
        self.assertEqual(
            q.encode(),
            (1 + 2 ** 127).to_bytes(length=16, byteorder='little'),
        )

        # Huge values.
        q = QuadFloat('1.1897e+4932')  # should be within range.
        self.assertTrue(q.is_finite())

        q = QuadFloat('1.1898e+4932')  # just overflows the range.
        self.assertTrue(q.is_infinite())

        # Infinities
        q = QuadFloat('Inf')
        self.assertTrue(q.is_infinite())
        self.assertFalse(q.is_sign_minus())

        q = QuadFloat('infinity')
        self.assertTrue(q.is_infinite())
        self.assertFalse(q.is_sign_minus())

        q = QuadFloat('-inf')
        self.assertTrue(q.is_infinite())
        self.assertTrue(q.is_sign_minus())

        q = QuadFloat('-INFINITY')
        self.assertTrue(q.is_infinite())
        self.assertTrue(q.is_sign_minus())

        # Nans with and without payloads
        for nan_string in ['nan', 'NaN', 'NAN', 'nAN', 'nan(1)', 'nan(9999)']:
            for prefix in '+', '-', '':
                q = QuadFloat(prefix + nan_string)
                self.assertTrue(q.is_nan())
                self.assertFalse(q.is_signaling())

            for prefix in '+', '-', '':
                q = QuadFloat(prefix + 's' + nan_string)
                self.assertTrue(q.is_nan())
                self.assertTrue(q.is_signaling())


        # Out-of-range payloads should just be clipped to be within range.
        q = QuadFloat('nan(123123123123123123123123123123123123)')

        with self.assertRaises(ValueError):
            QuadFloat('nan()')

        with self.assertRaises(ValueError):
            QuadFloat('+nan()')

        with self.assertRaises(ValueError):
            QuadFloat('+snan(1')

    def test_is_finite(self):
        self.assertTrue(QuadFloat('0.0').is_finite())
        self.assertTrue(QuadFloat('-0.0').is_finite())
        self.assertTrue(QuadFloat('8e-4933').is_finite())
        self.assertTrue(QuadFloat('-8e-4933').is_finite())
        self.assertTrue(QuadFloat('2.3').is_finite())
        self.assertTrue(QuadFloat('-2.3').is_finite())
        self.assertFalse(QuadFloat('Infinity').is_finite())
        self.assertFalse(QuadFloat('-Infinity').is_finite())
        self.assertFalse(QuadFloat('NaN').is_finite())
        self.assertFalse(QuadFloat('-NaN').is_finite())
        self.assertFalse(QuadFloat('sNaN').is_finite())
        self.assertFalse(QuadFloat('-sNaN').is_finite())

    def test_is_subnormal(self):
        self.assertFalse(QuadFloat('0.0').is_subnormal())
        self.assertFalse(QuadFloat('-0.0').is_subnormal())
        self.assertTrue(QuadFloat('3.3e-4932').is_subnormal())
        self.assertTrue(QuadFloat('-3.3e-4932').is_subnormal())
        self.assertFalse(QuadFloat('3.4e-4932').is_subnormal())
        self.assertFalse(QuadFloat('-3.4e-4932').is_subnormal())
        self.assertFalse(QuadFloat('2.3').is_subnormal())
        self.assertFalse(QuadFloat('-2.3').is_subnormal())
        self.assertFalse(QuadFloat('Infinity').is_subnormal())
        self.assertFalse(QuadFloat('-Infinity').is_subnormal())
        self.assertFalse(QuadFloat('NaN').is_subnormal())
        self.assertFalse(QuadFloat('-NaN').is_subnormal())
        self.assertFalse(QuadFloat('sNaN').is_subnormal())
        self.assertFalse(QuadFloat('-sNaN').is_subnormal())

    def test_is_sign_minus(self):
        self.assertFalse(QuadFloat('0.0').is_sign_minus())
        self.assertTrue(QuadFloat('-0.0').is_sign_minus())
        self.assertFalse(QuadFloat('8e-4933').is_sign_minus())
        self.assertTrue(QuadFloat('-8e-4933').is_sign_minus())
        self.assertFalse(QuadFloat('2.3').is_sign_minus())
        self.assertTrue(QuadFloat('-2.3').is_sign_minus())
        self.assertFalse(QuadFloat('Infinity').is_sign_minus())
        self.assertTrue(QuadFloat('-Infinity').is_sign_minus())
        self.assertFalse(QuadFloat('NaN').is_sign_minus())
        self.assertTrue(QuadFloat('-NaN').is_sign_minus())
        self.assertFalse(QuadFloat('sNaN').is_sign_minus())
        self.assertTrue(QuadFloat('-sNaN').is_sign_minus())

    def test_is_infinite(self):
        self.assertFalse(QuadFloat('0.0').is_infinite())
        self.assertFalse(QuadFloat('-0.0').is_infinite())
        self.assertFalse(QuadFloat('8e-4933').is_infinite())
        self.assertFalse(QuadFloat('-8e-4933').is_infinite())
        self.assertFalse(QuadFloat('2.3').is_infinite())
        self.assertFalse(QuadFloat('-2.3').is_infinite())
        self.assertTrue(QuadFloat('Infinity').is_infinite())
        self.assertTrue(QuadFloat('-Infinity').is_infinite())
        self.assertFalse(QuadFloat('NaN').is_infinite())
        self.assertFalse(QuadFloat('-NaN').is_infinite())
        self.assertFalse(QuadFloat('sNaN').is_infinite())
        self.assertFalse(QuadFloat('-sNaN').is_infinite())

    def test_is_nan(self):
        self.assertFalse(QuadFloat('0.0').is_nan())
        self.assertFalse(QuadFloat('-0.0').is_nan())
        self.assertFalse(QuadFloat('8e-4933').is_nan())
        self.assertFalse(QuadFloat('-8e-4933').is_nan())
        self.assertFalse(QuadFloat('2.3').is_nan())
        self.assertFalse(QuadFloat('-2.3').is_nan())
        self.assertFalse(QuadFloat('Infinity').is_nan())
        self.assertFalse(QuadFloat('-Infinity').is_nan())
        self.assertTrue(QuadFloat('NaN').is_nan())
        self.assertTrue(QuadFloat('-NaN').is_nan())
        self.assertTrue(QuadFloat('sNaN').is_nan())
        self.assertTrue(QuadFloat('-sNaN').is_nan())

    def test_is_signaling(self):
        self.assertFalse(QuadFloat('0.0').is_signaling())
        self.assertFalse(QuadFloat('-0.0').is_signaling())
        self.assertFalse(QuadFloat('8e-4933').is_signaling())
        self.assertFalse(QuadFloat('-8e-4933').is_signaling())
        self.assertFalse(QuadFloat('2.3').is_signaling())
        self.assertFalse(QuadFloat('-2.3').is_signaling())
        self.assertFalse(QuadFloat('Infinity').is_signaling())
        self.assertFalse(QuadFloat('-Infinity').is_signaling())
        self.assertFalse(QuadFloat('NaN').is_signaling())
        self.assertFalse(QuadFloat('-NaN').is_signaling())
        self.assertTrue(QuadFloat('sNaN').is_signaling())
        self.assertTrue(QuadFloat('-sNaN').is_signaling())

    def test_is_zero(self):
        self.assertTrue(QuadFloat('0.0').is_zero())
        self.assertTrue(QuadFloat('-0.0').is_zero())
        self.assertFalse(QuadFloat('8e-4933').is_zero())
        self.assertFalse(QuadFloat('-8e-4933').is_zero())
        self.assertFalse(QuadFloat('2.3').is_zero())
        self.assertFalse(QuadFloat('-2.3').is_zero())
        self.assertFalse(QuadFloat('Infinity').is_zero())
        self.assertFalse(QuadFloat('-Infinity').is_zero())
        self.assertFalse(QuadFloat('NaN').is_zero())
        self.assertFalse(QuadFloat('-NaN').is_zero())
        self.assertFalse(QuadFloat('sNaN').is_zero())
        self.assertFalse(QuadFloat('-sNaN').is_zero())

    def test_encode(self):
        test_values = [
            (QuadFloat(0), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'),
            (QuadFloat(1), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x3f'),
            (QuadFloat(2), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x40'),
            (QuadFloat(-1), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xbf'),
            (QuadFloat(-2), b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0'),
        ]
        for quad, expected in test_values:
            actual = quad.encode()
            self.assertEqual(
                actual,
                expected,
            )

    def test_encode_decode_roundtrip(self):
        test_values = [
            QuadFloat(0),
            QuadFloat(1),
            QuadFloat(-1),
            QuadFloat.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x7f'),  # inf
            QuadFloat.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff'),  # -inf
            QuadFloat.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\x7f'),  # qnan
            QuadFloat.decode(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\xff'),  # qnan
            QuadFloat.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\x7f'),  # qnan
            QuadFloat.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xff\xff'),  # qnan
            QuadFloat.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\x7f'),  # snan
            QuadFloat.decode(b'\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff'),  # snan
            QuadFloat('inf'),
            QuadFloat('-inf'),
            QuadFloat('nan'),
            QuadFloat('-nan'),
            QuadFloat('snan'),
            QuadFloat('-snan'),
        ]
        for value in test_values:
            encoded_value = value.encode()
            self.assertIsInstance(encoded_value, bytes)
            decoded_value = QuadFloat.decode(encoded_value)
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
            decoded_value = QuadFloat.decode(value)
            encoded_value = decoded_value.encode()

            self.assertIsInstance(encoded_value, bytes)
            self.assertEqual(value, encoded_value)

    def test_repr_construct_roundtrip(self):
        test_values = [
            QuadFloat('3.2'),
            QuadFloat(3.2),
            QuadFloat(1),
            QuadFloat('-1.0'),
            QuadFloat('0.0'),
            QuadFloat('-0.0'),
            QuadFloat('3.1415926535897932384626433'),
            QuadFloat('0.1'),
            QuadFloat('0.01'),
            QuadFloat('1e1000'),
            QuadFloat('1e-1000'),
            QuadFloat(0.10000000000001e-150),
            QuadFloat(0.32e-150),
            QuadFloat(0.99999999999999e-150),
            QuadFloat(0.10000000000001e-2),
            QuadFloat(0.32e-2),
            QuadFloat(0.99999999999999e-2),
            QuadFloat(0.10000000000001e-1),
            QuadFloat(0.32e-1),
            QuadFloat(0.99999999999999e-1),
            QuadFloat(0.10000000000001),
            QuadFloat(0.32),
            QuadFloat(0.99999999999999),
            QuadFloat(1),
            QuadFloat(3.2),
            QuadFloat(9.999999999999),
            QuadFloat(10.0),
            QuadFloat(10.00000000000001),
            QuadFloat(32),
            QuadFloat(0.10000000000001e150),
            QuadFloat(0.32e150),
            QuadFloat(0.99999999999999e150),
            QuadFloat(10**200),
            QuadFloat('inf'),
            QuadFloat('-inf'),
            QuadFloat('nan'),
            QuadFloat('-nan'),
            QuadFloat('snan'),
            QuadFloat('-snan'),
            QuadFloat('nan(123)'),
            QuadFloat('-snan(999999)'),
        ]
        for value in test_values:
            repr_value = repr(value)
            reconstructed_value = eval(repr_value)
            self.assertInterchangeable(value, reconstructed_value)

            str_value = str(value)
            reconstructed_value = QuadFloat(str_value)
            self.assertInterchangeable(value, reconstructed_value)

    def test_multiplication(self):
        # First steps
        a = QuadFloat(2)
        b = QuadFloat('inf')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('inf'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('inf'))

        a = QuadFloat(-2)
        b = QuadFloat('inf')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-inf'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-inf'))

        a = QuadFloat(2)
        b = QuadFloat('-inf')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-inf'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-inf'))

        a = QuadFloat(-2)
        b = QuadFloat('-inf')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('inf'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('inf'))

        a = QuadFloat('0.0')
        b = QuadFloat('inf')
        self.assertTrue((QuadFloat.multiplication(a, b)).is_nan())
        self.assertTrue((QuadFloat.multiplication(b, a)).is_nan())

        a = QuadFloat('0.0')
        b = QuadFloat('0.0')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('0.0'))

        a = QuadFloat('-0.0')
        b = QuadFloat('0.0')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-0.0'))

        a = QuadFloat('0.0')
        b = QuadFloat('-0.0')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-0.0'))

        a = QuadFloat('-0.0')
        b = QuadFloat('-0.0')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('0.0'))

        a = QuadFloat('2.0')
        b = QuadFloat('0.0')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('0.0'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('0.0'))

        a = QuadFloat('-2.0')
        b = QuadFloat('0.0')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-0.0'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-0.0'))

        a = QuadFloat('2.0')
        b = QuadFloat('-0.0')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-0.0'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-0.0'))

        a = QuadFloat('-2.0')
        b = QuadFloat('-0.0')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('0.0'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('0.0'))

        a = QuadFloat('2.0')
        b = QuadFloat('3.0')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('6.0'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('6.0'))

        # signaling nans?
        a = QuadFloat('-snan(123)')
        b = QuadFloat('2.3')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-nan(123)'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-nan(123)'))

        a = QuadFloat('-snan(123)')
        b = QuadFloat('nan(456)')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-nan(123)'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-nan(123)'))

        a = QuadFloat('-snan(123)')
        b = QuadFloat('-inf')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-nan(123)'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-nan(123)'))

        a = QuadFloat('-snan(123)')
        b = QuadFloat('-2.3')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-nan(123)'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-nan(123)'))

        # first snan wins
        a = QuadFloat('snan(123)')
        b = QuadFloat('-snan(456)')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('nan(123)'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-nan(456)'))

        # quiet nans with payload
        a = QuadFloat('2.0')
        b = QuadFloat('nan(789)')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('nan(789)'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('nan(789)'))

        a = QuadFloat('-2.0')
        b = QuadFloat('nan(789)')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-nan(789)'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-nan(789)'))

        a = QuadFloat('inf')
        b = QuadFloat('nan(789)')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('nan(789)'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('nan(789)'))

        a = QuadFloat('-inf')
        b = QuadFloat('nan(789)')
        self.assertInterchangeable(QuadFloat.multiplication(a, b), QuadFloat('-nan(789)'))
        self.assertInterchangeable(QuadFloat.multiplication(b, a), QuadFloat('-nan(789)'))

    def test_addition(self):
        # Cases where zeros are involved.
        a = QuadFloat('0.0')
        b = QuadFloat('0.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('0.0'))

        a = QuadFloat('0.0')
        b = QuadFloat('-0.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('0.0'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('0.0'))

        a = QuadFloat('-0.0')
        b = QuadFloat('-0.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('-0.0'))

        a = QuadFloat('2.0')
        b = QuadFloat('0.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('2.0'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('2.0'))

        a = QuadFloat('2.0')
        b = QuadFloat('-2.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('0.0'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('0.0'))

        a = QuadFloat('2.0')
        b = QuadFloat('3.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('5.0'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('5.0'))

        # Infinities.
        a = QuadFloat('inf')
        b = QuadFloat('2.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('inf'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('inf'))

        a = QuadFloat('inf')
        b = QuadFloat('-2.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('inf'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('inf'))

        a = QuadFloat('-inf')
        b = QuadFloat('2.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('-inf'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('-inf'))

        a = QuadFloat('-inf')
        b = QuadFloat('-2.0')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('-inf'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('-inf'))

        a = QuadFloat('-inf')
        b = QuadFloat('inf')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('nan'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('nan'))

        a = QuadFloat('inf')
        b = QuadFloat('inf')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('inf'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('inf'))

        a = QuadFloat('-inf')
        b = QuadFloat('-inf')
        self.assertInterchangeable(QuadFloat.addition(a, b), QuadFloat('-inf'))
        self.assertInterchangeable(QuadFloat.addition(b, a), QuadFloat('-inf'))

    def test_subtraction(self):
        # XXX Needs some tests!
        # Pay particular attention to handling of NaNs:  subtraction is *not* the
        # same as addition with the second argument negated if nans are present.
        pass

    def test_division(self):
        # Finite: check all combinations of signs.
        a = QuadFloat('1.0')
        b = QuadFloat('2.0')
        self.assertInterchangeable(QuadFloat.division(a, b), QuadFloat('0.5'))
        self.assertInterchangeable(QuadFloat.division(b, a), QuadFloat('2.0'))

        a = QuadFloat('-1.0')
        b = QuadFloat('2.0')
        self.assertInterchangeable(QuadFloat.division(a, b), QuadFloat('-0.5'))
        self.assertInterchangeable(QuadFloat.division(b, a), QuadFloat('-2.0'))

        a = QuadFloat('1.0')
        b = QuadFloat('-2.0')
        self.assertInterchangeable(QuadFloat.division(a, b), QuadFloat('-0.5'))
        self.assertInterchangeable(QuadFloat.division(b, a), QuadFloat('-2.0'))

        a = QuadFloat('-1.0')
        b = QuadFloat('-2.0')
        self.assertInterchangeable(QuadFloat.division(a, b), QuadFloat('0.5'))
        self.assertInterchangeable(QuadFloat.division(b, a), QuadFloat('2.0'))

        # One or other argument zero (but not both).
        a = QuadFloat('0.0')
        b = QuadFloat('2.0')
        self.assertInterchangeable(QuadFloat.division(a, b), QuadFloat('0.0'))
        self.assertInterchangeable(QuadFloat.division(b, a), QuadFloat('inf'))

        a = QuadFloat('0.0')
        b = QuadFloat('-2.0')
        self.assertInterchangeable(QuadFloat.division(a, b), QuadFloat('-0.0'))
        self.assertInterchangeable(QuadFloat.division(b, a), QuadFloat('-inf'))

        a = QuadFloat('-0.0')
        b = QuadFloat('2.0')
        self.assertInterchangeable(QuadFloat.division(a, b), QuadFloat('-0.0'))
        self.assertInterchangeable(QuadFloat.division(b, a), QuadFloat('-inf'))

        a = QuadFloat('-0.0')
        b = QuadFloat('-2.0')
        self.assertInterchangeable(QuadFloat.division(a, b), QuadFloat('0.0'))
        self.assertInterchangeable(QuadFloat.division(b, a), QuadFloat('inf'))

        # Zero divided by zero.
        a = QuadFloat('0.0')
        b = QuadFloat('0.0')
        self.assertTrue(QuadFloat.division(a, b).is_nan())

        # XXX Tests for infinities and nans as inputs.
        # XXX Tests for correct rounding.
        # XXX Tests for subnormal results, underflow.

    def test_negate(self):
        self.assertInterchangeable(-QuadFloat('-2.0'), QuadFloat('2.0'))
        self.assertInterchangeable(-QuadFloat('2.0'), QuadFloat('-2.0'))
        self.assertInterchangeable(-QuadFloat('-0.0'), QuadFloat('0.0'))
        self.assertInterchangeable(-QuadFloat('0.0'), QuadFloat('-0.0'))
        self.assertInterchangeable(-QuadFloat('-inf'), QuadFloat('inf'))
        self.assertInterchangeable(-QuadFloat('inf'), QuadFloat('-inf'))
        self.assertInterchangeable(-QuadFloat('-nan'), QuadFloat('nan'))
        self.assertInterchangeable(-QuadFloat('nan'), QuadFloat('-nan'))
        self.assertInterchangeable(-QuadFloat('-snan'), QuadFloat('snan'))
        self.assertInterchangeable(-QuadFloat('snan'), QuadFloat('-snan'))
        self.assertInterchangeable(-QuadFloat('-nan(123)'), QuadFloat('nan(123)'))
        self.assertInterchangeable(-QuadFloat('nan(123)'), QuadFloat('-nan(123)'))
        self.assertInterchangeable(-QuadFloat('-snan(123)'), QuadFloat('snan(123)'))
        self.assertInterchangeable(-QuadFloat('snan(123)'), QuadFloat('-snan(123)'))

    def test_abs(self):
        self.assertInterchangeable(abs(QuadFloat('-2.0')), QuadFloat('2.0'))
        self.assertInterchangeable(abs(QuadFloat('2.0')), QuadFloat('2.0'))
        self.assertInterchangeable(abs(QuadFloat('-0.0')), QuadFloat('0.0'))
        self.assertInterchangeable(abs(QuadFloat('0.0')), QuadFloat('0.0'))
        self.assertInterchangeable(abs(QuadFloat('-inf')), QuadFloat('inf'))
        self.assertInterchangeable(abs(QuadFloat('inf')), QuadFloat('inf'))
        self.assertInterchangeable(abs(QuadFloat('-nan')), QuadFloat('nan'))
        self.assertInterchangeable(abs(QuadFloat('nan')), QuadFloat('nan'))
        self.assertInterchangeable(abs(QuadFloat('-snan')), QuadFloat('snan'))
        self.assertInterchangeable(abs(QuadFloat('snan')), QuadFloat('snan'))
        self.assertInterchangeable(abs(QuadFloat('-nan(123)')), QuadFloat('nan(123)'))
        self.assertInterchangeable(abs(QuadFloat('nan(123)')), QuadFloat('nan(123)'))
        self.assertInterchangeable(abs(QuadFloat('-snan(123)')), QuadFloat('snan(123)'))
        self.assertInterchangeable(abs(QuadFloat('snan(123)')), QuadFloat('snan(123)'))


if __name__ == '__main__':
    unittest.main()

