import unittest

from binary_interchange_format import BinaryInterchangeFormat

Float16 = BinaryInterchangeFormat(width=16)


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


if __name__ == '__main__':
    unittest.main()
