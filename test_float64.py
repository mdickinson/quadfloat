"""
Random test for the Float64 type, comparing results with the hardware-generated
values.

"""

import math
import random
import struct
import unittest

from binary_interchange_format import BinaryInterchangeFormat


Float64 = BinaryInterchangeFormat(width=64)


class TestFloat64(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two Float128 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertTrue(quad1._equivalent(quad2),
                        msg = '{!r} not equivalent to {!r}'.format(quad1, quad2))

    def random_float(self):
        """
        Return a random hardware float, avoiding infinities and NaNs.

        """
        exponent_bits = random.randrange(2**11 - 1)
        sign_bit = random.randrange(2)
        significand_bits = random.randrange(2**52)
        equivalent_integer = ((sign_bit << 11) + exponent_bits << 52) + significand_bits
        return struct.unpack('<d', struct.pack('<Q', equivalent_integer))[0]

    def test_random_additions(self):
        for i in range(10000):
            x = self.random_float()
            y = self.random_float()

            # Float64 addition.
            result1 = Float64(x) + Float64(y)
            # float addition.
            result2 = Float64(x + y)
            self.assertInterchangeable(result1, result2)

    def test_random_subtractions(self):
        for i in range(10000):
            x = self.random_float()
            y = self.random_float()

            # Float64 addition.
            result1 = Float64(x) - Float64(y)
            # float addition.
            result2 = Float64(x - y)
            self.assertInterchangeable(result1, result2)

    def test_random_multiplications(self):
        for i in range(10000):
            x = self.random_float()
            y = self.random_float()

            # Float64 addition.
            result1 = Float64(x) - Float64(y)
            # float addition.
            result2 = Float64(x - y)
            self.assertInterchangeable(result1, result2)

    def test_random_divisions(self):
        for i in range(10000):
            x = self.random_float()
            y = self.random_float()

            # Float64 addition.
            result1 = Float64(x) - Float64(y)
            # float addition.
            result2 = Float64(x - y)
            self.assertInterchangeable(result1, result2)

    def test_random_sqrt(self):
        for i in range(10000):
            x = self.random_float()
            result1 = Float64.square_root(Float64(x))
            try:
                sqrtx = math.sqrt(x)
            except ValueError:
                sqrtx = float('nan')
            result2 = Float64(sqrtx)
            self.assertInterchangeable(result1, result2)


if __name__ == '__main__':
    unittest.main()
