import unittest

from binary_interchange_format import BinaryInterchangeFormat


Float16 = BinaryInterchangeFormat(width=16)
Float32 = BinaryInterchangeFormat(width=32)


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


if __name__ == '__main__':
    unittest.main()
