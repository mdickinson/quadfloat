import unittest


from quadfloat.api import BinaryInterchangeFormat, BitString
from quadfloat.parsing import parse_finite_decimal


class TestShortRepr(unittest.TestCase):
    def test_short_repr_subnormal_corner_case(self):
        # Test rarely-occurring corner case where the naive algorithm (which
        # establishes an exponent first and then gets the shortest string for
        # that exponent) can fail to give the 'best' shortest string.  This
        # corner case can only occur for small subnormals, and doesn't occur at
        # all for binary16, binary32, binary64 or binary128.

        # An example where this does occur is for binary256:  the 4th smallest
        # subnormal value is 2**-262376, or approx. 8.9920e-78984.  The
        # interval of values that rounds to this contains
        # (7.869e-78984, 10.11e-78984). The closest 1-digit value that works
        # is clearly 9e-78984, but the naive algorithm can come up with
        # 1e-78983 instead.

        # Smallest representable positive binary256 value (subnormal).
        binary256 = BinaryInterchangeFormat(256)
        TINY = binary256.decode(BitString.from_int(width=256, value_as_int=1))
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
                parse_finite_decimal(input_string),
                parse_finite_decimal(output_string),
            )
