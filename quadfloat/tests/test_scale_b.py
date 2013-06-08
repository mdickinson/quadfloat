import contextlib
import unittest

from quadfloat import binary16
from quadfloat.attributes import (
    inexact_handler,
    invalid_operation_handler,
    overflow_handler,
    underflow_handler,
)
from quadfloat.exceptions import OverflowException


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
# 'before rounding' definition (regardless of rounding direction).
#
# 'underflow' may be combined with 'inexact';  for 'overflow', the result is
# always inexact, and we leave it out since it's redundant.


# 16-bit test values, round-half-to-even.
test16 = """\
# NaNs behave as usual
nan 0 -> nan
-nan(123) 54 -> -nan(123)

# Infinities are unchanged...
inf -1000 -> inf
inf -1 -> inf
inf 0 -> inf
inf 1 -> inf
inf 1000 -> inf
-inf -1000 -> -inf
-inf -1 -> -inf
-inf 0 -> -inf
-inf 1 -> -inf
-inf 1000 -> -inf

# As are zeros.
0x0p0 -1000 -> 0x0p0
0x0p0 -1 -> 0x0p0
0x0p0 0 -> 0x0p0
0x0p0 1 -> 0x0p0
0x0p0 1000 -> 0x0p0
-0x0p0 -1000 -> -0x0p0
-0x0p0 -1 -> -0x0p0
-0x0p0 0 -> -0x0p0
-0x0p0 1 -> -0x0p0
-0x0p0 1000 -> -0x0p0

# Powers of 2.
0x1p0 -1000 -> 0x0p0 underflow inexact
0x1p0 -25 -> 0x0p0 underflow inexact
0x1p0 -24 -> 0x1p-24 underflow
0x1p0 -16 -> 0x1p-16 underflow
0x1p0 -15 -> 0x1p-15 underflow
0x1p0 -14 -> 0x1p-14
0x1p0 -3 -> 0x1p-3
0x1p0 -2 -> 0x1p-2
0x1p0 -1 -> 0x1p-1
0x1p0 0 -> 0x1p0
0x1p0 1 -> 0x1p1
0x1p0 2 -> 0x1p2
0x1p0 3 -> 0x1p3
0x1p0 15 -> 0x1p15
0x1p0 16 -> inf overflow
0x1p0 1000 -> inf overflow

# Check rounding cases with subnormal result.
0x1p-2 -24 -> 0x0p0 underflow inexact
0x2p-2 -24 -> 0x0p0 underflow inexact
0x3p-2 -24 -> 0x1p-24 underflow inexact
0x4p-2 -24 -> 0x1p-24 underflow
0x5p-2 -24 -> 0x1p-24 underflow inexact
0x6p-2 -24 -> 0x2p-24 underflow inexact
0x7p-2 -24 -> 0x2p-24 underflow inexact
0x8p-2 -24 -> 0x2p-24 underflow
0x9p-2 -24 -> 0x2p-24 underflow inexact
0xap-2 -24 -> 0x2p-24 underflow inexact
0xbp-2 -24 -> 0x3p-24 underflow inexact
0xcp-2 -24 -> 0x3p-24 underflow
0xdp-2 -24 -> 0x3p-24 underflow inexact
0xep-2 -24 -> 0x4p-24 underflow inexact
0xfp-2 -24 -> 0x4p-24 underflow inexact
0x10p-2 -24 -> 0x4p-24 underflow

"""


@contextlib.contextmanager
def catch_exceptions():
    signal_list = []

    def my_handler(exc):
        signal_list.append(exc)
        return exc.default_handler()

    with invalid_operation_handler(my_handler):
        with inexact_handler(my_handler):
            with overflow_handler(my_handler):
                with underflow_handler(my_handler):
                    yield signal_list


class TestScaleB(unittest.TestCase):
    def test_scale_b(self):
        arg_converters = binary16.convert_from_hex_character, int
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

            rhs = result.split()
            expected = binary16.convert_from_hex_character(rhs[0])
            flags = rhs[1:]

            with catch_exceptions() as exceptions:
                actual = source1.scale_b(exp)

            expect_overflow = 'overflow' in flags
            got_overflow = any(
                isinstance(exc, OverflowException) for exc in exceptions)
            self.assertEqual(expect_overflow, got_overflow)

            # Check that actual and expected are interchangeable.
            self.assertEqual(
                actual.format,
                expected.format,
            )
            self.assertEqual(
                binary16.convert_to_hex_character(actual),
                binary16.convert_to_hex_character(expected),
            )


if __name__ == '__main__':
    unittest.main()
