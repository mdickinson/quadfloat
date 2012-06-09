import decimal
import unittest


from binary_interchange_format import BinaryInterchangeFormat


class TestShortRepr(unittest.TestCase):
    def test_short_repr_subnormal_corner_case(self):
        # Test rarely-occurring corner case where the naive algorithm (which
        # establishes an exponent first and then gets the shortest string for
        # that exponent) can fail to give the 'best' shortest string.  This
        # corner case can only occur for small subnormals, and doesn't occur at
        # all for Float16, Float32, Float64 or Float128.

        # An example where this does occur is for Float256:  the 4th smallest
        # subnormal value is 2**-262376, or approx. 8.9920e-78984.  The interval
        # of values that rounds to this contains (7.869e-78984, 10.11e-78984).
        # The closest 1-digit value that works is clearly 9e-78984, but the
        # naive algorithm can come up with 1e-78983 instead.

        # Smallest representable positive Float256 value (subnormal).
        Float256 = BinaryInterchangeFormat(256)
        TINY = Float256.decode(b'\x01' + 31 * b'\x00')
        test_pairs = [
            (TINY, '2e-78984'),
            (2 * TINY, '4e-78984'),
            (3 * TINY, '7e-78984'),
            (4 * TINY, '9e-78984'),   # <-- here's the problematic value:
        ]
        for input, output_string in test_pairs:
            # XXX Conversion is slow!
            input_string = str(input)
            self.assertEqual(
                decimal.Decimal(input_string),
                decimal.Decimal(output_string),
            )


if __name__ == '__main__':
    unittest.main()
    