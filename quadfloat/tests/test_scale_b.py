import unittest

from quadfloat.tests.parse_test_data import parse_test_data


test_data = """\
attribute rounding-direction: roundTiesToEven
attribute tininess-detection: afterRounding

operation: scaleB
operation source: binary16

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
0x1p0 16 -> inf inexact overflow
0x1p0 1000 -> inf inexact overflow

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


class TestScaleB(unittest.TestCase):
    def test_data(self):
        for test_case in parse_test_data(test_data):
            if test_case.actual_result != test_case.expected_result:
                self.fail(
                    "Error in test case:\n{0!r}".format(test_case)
                )
