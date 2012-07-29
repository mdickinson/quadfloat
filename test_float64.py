"""
Random test for the float64 type, comparing results with the hardware-generated
values.

"""

import math
import random
import struct
import unittest

from binary_interchange_format import BinaryInterchangeFormat

from binary_interchange_format import _FINITE, _INFINITE, _NAN


float64 = BinaryInterchangeFormat(width=64)


class TestFloat64(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two float128 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
    def assertInterchangeable(self, quad1, quad2, msg = ''):
        """
        Assert that two float16 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        # XXX Digs into private details, which isn't ideal.
        if quad1._type != quad2._type:
            interchangeable = False
        elif quad1._type == _FINITE:
            interchangeable = (
                quad1._sign == quad2._sign and
                quad1._exponent == quad2._exponent and
                quad1._significand == quad2._significand
            )
        elif quad1._type == _INFINITE:
            interchangeable = quad1._sign == quad2._sign
        elif quad1._type == _NAN:
            interchangeable = (
                quad1._sign == quad2._sign and
                quad1._signaling == quad2._signaling and
                quad1._payload == quad2._payload
            )
        else:
            assert False, "never get here"

        self.assertTrue(interchangeable,
                        msg = msg + '{!r} not interchangeable with {!r}'.format(quad1, quad2))

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

            # float64 addition.
            result1 = float64(x) + float64(y)
            # float addition.
            result2 = float64(x + y)
            self.assertInterchangeable(result1, result2)

    def test_random_subtractions(self):
        for i in range(10000):
            x = self.random_float()
            y = self.random_float()

            # float64 addition.
            result1 = float64(x) - float64(y)
            # float addition.
            result2 = float64(x - y)
            self.assertInterchangeable(result1, result2)

    def test_random_multiplications(self):
        for i in range(10000):
            x = self.random_float()
            y = self.random_float()

            # float64 addition.
            result1 = float64(x) - float64(y)
            # float addition.
            result2 = float64(x - y)
            self.assertInterchangeable(result1, result2)

    def test_random_divisions(self):
        for i in range(10000):
            x = self.random_float()
            y = self.random_float()

            # float64 addition.
            result1 = float64(x) - float64(y)
            # float addition.
            result2 = float64(x - y)
            self.assertInterchangeable(result1, result2)

    def test_random_sqrt(self):
        for i in range(10000):
            x = self.random_float()
            result1 = float64.square_root(float64(x))
            try:
                sqrtx = math.sqrt(x)
            except ValueError:
                sqrtx = float('nan')
            result2 = float64(sqrtx)
            self.assertInterchangeable(result1, result2)


if __name__ == '__main__':
    unittest.main()
