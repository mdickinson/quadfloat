import unittest

from quadfloat.binary_interchange_format import BinaryInterchangeFormat


# Format notes:
#
# Possible flags.
#
#  overflow
#  underflow
#  underflow-before --- indicates that underflow occurs only if we're using the
#    'before rounding' rules
#  inexact
#  invalid-operation
#  division-by-zero
#
# There's no need for an underflow-after flag: if we've got underflow according
# to the 'after rounding' definition, we've also got underflow according to the
# 'before rounding' definition (regardless of rounding mode).
#
# 'underflow' may be combined with 'inexact';  for 'overflow', the result is
# always inexact, and we leave it out since it's redundant.


# 16-bit test values, round-half-to-even.
test16 = """\
0x1p0 -25 -> 0x0p0 underflow
0x1p0 -24 -> 0x1p-24 underflow
0x1.8p0 -24 -> 0x2p-24 underflow inexact
0x1p0 -15 -> 0x1p-15
0x1p0 -14 -> 0x1p-14
0x1p0 15 -> 0x8000p0
0x1p0 16 -> Infinity overflow
"""


float16 = BinaryInterchangeFormat(width=16)


class TestScaleB(unittest.TestCase):
    def test_scale_b(self):
        arg_converters = float16.convert_from_hex_character, int
        for line in test16.splitlines():
            # Strip comments.
            if '#' in line:
                line = line[:line.index('#')]

            # Skip empty lines, or lines containing
            # only whitespace and/or comments.
            if not line.strip():
                continue

            args, result = line.split('->')
            args = args.split()

            assert len(args) == len(arg_converters)
            args = [
                converter(arg)
                for converter, arg in zip(arg_converters, args)
            ]
            source1, exp = args

            # XXX Do something with the flags.
            rhs = result.split()
            expected = float16.convert_from_hex_character(rhs[0])
            flags = rhs[1:]

            actual = source1.scale_b(exp)

            # Check that actual and expected are interchangeable.
            self.assertEqual(
                actual.format,
                expected.format,
            )
            self.assertEqual(
                float16.convert_to_hex_character(actual),
                float16.convert_to_hex_character(expected),
            )


if __name__ == '__main__':
    unittest.main()
