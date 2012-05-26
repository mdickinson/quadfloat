import unittest

from binary_interchange_format import BinaryInterchangeFormat


Float16 = BinaryInterchangeFormat(width=16)
Float32 = BinaryInterchangeFormat(width=32)
Float64 = BinaryInterchangeFormat(width=64)


class TestFloat16(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two Float128 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertTrue(quad1._equivalent(quad2),
                        msg = '{!r} not equivalent to {!r}'.format(quad1, quad2))

    def test_construction_from_float(self):
        actual = Float16(0.9)
        expected = Float16('0.89990234375')
        self.assertInterchangeable(actual, expected)

    def test_construction_from_int(self):
        # Test round-half-to-even

        # 2048 -> significand bits of 0, exponent of ???
        # 5 exponent bits;  for 1.0, would expect exponent bits to have value 15
        # so for 2048.0, should be 15+ 11 = 26.  Shift by 2 to get 104.
        self.assertEqual(Float16(2048).encode(), b'\x00\x68')
        self.assertEqual(Float16(2049).encode(), b'\x00\x68')
        self.assertEqual(Float16(2050).encode(), b'\x01\x68')
        self.assertEqual(Float16(2051).encode(), b'\x02\x68')
        self.assertEqual(Float16(2052).encode(), b'\x02\x68')
        self.assertEqual(Float16(2053).encode(), b'\x02\x68')
        self.assertEqual(Float16(2054).encode(), b'\x03\x68')
        self.assertEqual(Float16(2055).encode(), b'\x04\x68')
        self.assertEqual(Float16(2056).encode(), b'\x04\x68')

    def test_construction_from_float(self):
        # Test round-half-to-even

        self.assertEqual(Float16(2048.0).encode(), b'\x00\x68')
        # halfway case
        self.assertEqual(Float16(2048.9999999999).encode(), b'\x00\x68')
        self.assertEqual(Float16(2049.0).encode(), b'\x00\x68')
        self.assertEqual(Float16(2049.0000000001).encode(), b'\x01\x68')
        self.assertEqual(Float16(2050.0).encode(), b'\x01\x68')
        self.assertEqual(Float16(2050.9999999999).encode(), b'\x01\x68')
        self.assertEqual(Float16(2051.0).encode(), b'\x02\x68')
        self.assertEqual(Float16(2051.0000000001).encode(), b'\x02\x68')
        self.assertEqual(Float16(2052.0).encode(), b'\x02\x68')
        self.assertEqual(Float16(2053.0).encode(), b'\x02\x68')
        self.assertEqual(Float16(2054.0).encode(), b'\x03\x68')
        self.assertEqual(Float16(2055.0).encode(), b'\x04\x68')
        self.assertEqual(Float16(2056.0).encode(), b'\x04\x68')

        # Subnormals.
        eps = 1e-10
        tiny = 2.0**-24  # smallest positive representable float16 subnormal
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
            self.assertEqual(Float16(x).encode(), bs)

    def test_division(self):
        # Test division, with particular attention to correct rounding.
        # Integers up to 2048 all representable in Float16

        self.assertInterchangeable(
            Float16.division(Float32(2048*4), Float32(4)),
            Float16(2048)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 1), Float32(4)),
            Float16(2048)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 2), Float32(4)),
            Float16(2048)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 3), Float32(4)),
            Float16(2048)
        )
        # Exact halfway case; should be rounded down.
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 4), Float32(4)),
            Float16(2048)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 5), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 6), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 7), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 8), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 9), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 10), Float32(4)),
            Float16(2050)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 11), Float32(4)),
            Float16(2050)
        )
        # Exact halfway case, rounds *up*!
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 12), Float32(4)),
            Float16(2052)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 13), Float32(4)),
            Float16(2052)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 14), Float32(4)),
            Float16(2052)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 15), Float32(4)),
            Float16(2052)
        )
        self.assertInterchangeable(
            Float16.division(Float32(2048*4 + 16), Float32(4)),
            Float16(2052)
        )

    def test_sqrt(self):
        # Easy small integer cases.
        for i in range(1, 46):
            self.assertInterchangeable(
                Float16.square_root(Float16(i * i)),
                Float16(i),
            )

        # Zeros.
        self.assertInterchangeable(
            Float16.square_root(Float16('0')),
            Float16('0'),
        )
        self.assertInterchangeable(
            Float16.square_root(Float16('-0')),
            Float16('-0'),
        )

        # Infinities
        self.assertInterchangeable(
            Float16.square_root(Float16('inf')),
            Float16('inf'),
        )

        self.assertInterchangeable(
            Float16.square_root(Float16('-inf')),
            Float16('nan'),
        )

        # Negatives
        self.assertInterchangeable(
            Float16.square_root(Float16('-4.0')),
            Float16('nan'),
        )

        # NaNs
        self.assertInterchangeable(
            Float16.square_root(Float16('snan(456)')),
            Float16('nan(456)'),
        )

        self.assertInterchangeable(
            Float16.square_root(Float16('-nan(123)')),
            Float16('-nan(123)'),
        )

        # Subnormal results.
        tiny = 2.0**-24  # smallest positive representable float16 subnormal

        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny)),
            Float16(tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 0.25)),
            Float16('0.0'),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 0.250000001)),
            Float16(tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny)),
            Float16(tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 2.24999999999)),
            Float16(tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 2.25)),
            Float16(2 * tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 2.250000001)),
            Float16(2 * tiny),
        )
        self.assertInterchangeable(
            Float16.square_root(Float64(tiny * tiny * 4.0)),
            Float16(2 * tiny),
        )


if __name__ == '__main__':
    unittest.main()
