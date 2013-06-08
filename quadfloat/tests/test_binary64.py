"""
Random test for the binary64 type, comparing results with the
hardware-generated values.

"""

import math
import random
import struct
import unittest

from quadfloat.binary_interchange_format import BinaryInterchangeFormat
from quadfloat.tests.base_test_case import BaseTestCase


binary64 = BinaryInterchangeFormat(width=64)


class TestBinary64(BaseTestCase):
    def random_finite_float(self):
        """
        Return a random hardware float, avoiding infinities and NaNs.

        """
        sign = random.randrange(2) << 63
        exponent = random.randrange(2 ** 11 - 1) << 52
        significand = random.randrange(2 ** 52)
        equivalent_integer = sign + exponent + significand
        return struct.unpack('<d', struct.pack('<Q', equivalent_integer))[0]

    def test_random_additions(self):
        for i in range(10000):
            x = self.random_finite_float()
            y = self.random_finite_float()

            # binary64 addition.
            result1 = binary64(x) + binary64(y)
            # float addition.
            result2 = binary64(x + y)
            self.assertInterchangeable(result1, result2)

    def test_random_subtractions(self):
        for i in range(10000):
            x = self.random_finite_float()
            y = self.random_finite_float()

            # binary64 addition.
            result1 = binary64(x) - binary64(y)
            # float addition.
            result2 = binary64(x - y)
            self.assertInterchangeable(result1, result2)

    def test_random_multiplications(self):
        for i in range(10000):
            x = self.random_finite_float()
            y = self.random_finite_float()

            # binary64 addition.
            result1 = binary64(x) - binary64(y)
            # float addition.
            result2 = binary64(x - y)
            self.assertInterchangeable(result1, result2)

    def test_random_divisions(self):
        for i in range(10000):
            x = self.random_finite_float()
            y = self.random_finite_float()

            # binary64 addition.
            result1 = binary64(x) - binary64(y)
            # float addition.
            result2 = binary64(x - y)
            self.assertInterchangeable(result1, result2)

    def test_random_sqrt(self):
        for i in range(10000):
            x = self.random_finite_float()
            result1 = binary64.square_root(binary64(x))
            try:
                sqrtx = math.sqrt(x)
            except ValueError:
                sqrtx = float('nan')
            result2 = binary64(sqrtx)
            self.assertInterchangeable(result1, result2)
