"""
Tests for mixed-precision arithmetic.

"""

import unittest

from binary_interchange_format import BinaryInterchangeFormat


Float16 = BinaryInterchangeFormat(width=16)
Float32 = BinaryInterchangeFormat(width=32)


class TestMixed(unittest.TestCase):
    def assertInterchangeable(self, quad1, quad2):
        """
        Assert that two Float128 instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertTrue(quad1._equivalent(quad2),
                        msg = '{!r} not equivalent to {!r}'.format(quad1, quad2))

    def test_nan_payload(self):
        # a Float32 NaN has 22 bits devoted to payload; Float16 has only 9.
        # XXX Should really make _payload public if we're going to test it this way.
        source1 = Float32('nan')
        self.assertEqual(source1._payload, 0)

        source1 = Float32('nan(0)')
        self.assertEqual(source1._payload, 0)
        
        # Payloads larger than that allowed should be clipped to be within range.
        # XXX As an implementation choice, this needs documenting.
        source1 = Float32('nan({})'.format(2**22))
        self.assertEqual(source1._payload, 2**22 - 1)

        source1 = Float32('snan(0)')
        self.assertEqual(source1._payload, 1)

        source1 = Float32('snan({})'.format(2**22))
        self.assertEqual(source1._payload, 2**22 - 1)

        source1 = Float16('snan({})'.format(2**22))
        self.assertEqual(source1._payload, 2**9 - 1)

        # Now combine two Float32 instances with a Float16 result; NaN should be shortened
        # appropriately.
        source1 = Float32('nan(999999)')
        source2 = Float32('2.0')
        for op in Float16.addition, Float16.subtraction, Float16.multiplication, Float16.division:
            result = op(source1, source2)
            self.assertEqual(result._format, Float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, False)

            result = op(source2, source1)
            self.assertEqual(result._format, Float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, False)

        source1 = Float32('-snan(999999)')
        source2 = Float32('2.0')
        for op in Float16.addition, Float16.subtraction, Float16.multiplication, Float16.division:
            result = op(source1, source2)
            self.assertEqual(result._format, Float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, True)

            result = op(source2, source1)
            self.assertEqual(result._format, Float16)
            self.assertEqual(result._payload, 2**9 - 1)
            self.assertEqual(result._sign, True)


if __name__ == '__main__':
    unittest.main()
