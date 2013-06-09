import unittest

from quadfloat.tests.parse_test_data import parse_test_data


FAILURE_MSG_TEMPLATE = """\
Flags don't match for failed test: {0!r}
Actual flags: {1!r}
Expected flags: {2!r}
"""


test_data = """\
operation: roundToIntegralTiesToAway
operation source: binary16

# Integers.
-0x3p0 -> -0x3p0
-0x2p0 -> -0x2p0
-0x1p0 -> -0x1p0
-0x0p0 -> -0x0p0  # sign preserved on zeros.
0x0p0 -> 0x0p0
0x1p0 -> 0x1p0
0x2p0 -> 0x2p0
0x3p0 -> 0x3p0

-0x3.8p0 -> -0x4p0
-0x2.8p0 -> -0x3p0
-0x1.8p0 -> -0x2p0
-0x0.8p0 -> -0x1p0
0x0.8p0 -> 0x1p0
0x1.8p0 -> 0x2p0
0x2.8p0 -> 0x3p0
0x3.8p0 -> 0x4p0

-0x3.4p0 -> -0x3p0
-0x2.4p0 -> -0x2p0
-0x1.4p0 -> -0x1p0
-0x0.4p0 -> -0x0p0
0x0.4p0 -> 0x0p0
0x1.4p0 -> 0x1p0
0x2.4p0 -> 0x2p0
0x3.4p0 -> 0x3p0

-0x3.cp0 -> -0x4p0
-0x2.cp0 -> -0x3p0
-0x1.cp0 -> -0x2p0
-0x0.cp0 -> -0x1p0
0x0.cp0 -> 0x1p0
0x1.cp0 -> 0x2p0
0x2.cp0 -> 0x3p0
0x3.cp0 -> 0x4p0

# Infinities and quiet NaNs
-Infinity -> -Infinity
Infinity -> Infinity
NaN -> NaN
-NaN(123) -> -NaN(123)

# Signaling NaNs
sNaN -> NaN(1) invalid
-sNaN(123) -> -NaN(123) invalid

"""


class TestRoundToIntegral(unittest.TestCase):
    def test_data(self):
        for test_case in parse_test_data(test_data):
            if test_case.actual_result != test_case.expected_result:
                self.fail(
                    "Error in test case:\n{0!r}".format(test_case)
                )
