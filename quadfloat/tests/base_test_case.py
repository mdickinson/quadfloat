import unittest

from quadfloat.api import encode


def _identifying_string(binary_float):
    return "{0} (format {1})".format(
        encode(binary_float),
        binary_float.format,
    )


class BaseTestCase(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2, msg=None):
        """
        Assert that two _BinaryFloat instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertEqual(
            _identifying_string(quad1),
            _identifying_string(quad2),
            msg)

    def assertNotInterchangeable(self, quad1, quad2, msg=None):
        """
        Assert that two _BinaryFloat instances are not interchangeable.

        See also: assertInterchangeable

        """
        self.assertNotEqual(
            _identifying_string(quad1),
            _identifying_string(quad2),
            msg)
