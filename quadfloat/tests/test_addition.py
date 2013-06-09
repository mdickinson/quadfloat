import unittest

from quadfloat.tests.parse_test_data import parse_test_data


test_data = """\
attribute rounding-direction: roundTiesToEven
attribute tininess-detection: afterRounding

operation: addition
operation destination: binary16
operation source1: binary16
operation source2: binary16

# Zeros
0x0p0 0x0p0 -> 0x0p0
-0x0p0 -0x0p0 -> -0x0p0
0x0p0 -0x0p0 -> 0x0p0
-0x0p0 0x0p0 -> 0x0p0
0x1p0 -0x1p0 -> 0x0p0

# Infinities
Infinity Infinity -> Infinity
-Infinity -Infinity -> -Infinity
Infinity -Infinity -> NaN invalid
-Infinity Infinity -> NaN invalid

attribute rounding-direction: roundTiesToAway

# Zeros
0x0p0 0x0p0 -> 0x0p0
-0x0p0 -0x0p0 -> -0x0p0
0x0p0 -0x0p0 -> 0x0p0
-0x0p0 0x0p0 -> 0x0p0
0x1p0 -0x1p0 -> 0x0p0

attribute rounding-direction: roundTowardPositive

# Zeros
0x0p0 0x0p0 -> 0x0p0
-0x0p0 -0x0p0 -> -0x0p0
0x0p0 -0x0p0 -> 0x0p0
-0x0p0 0x0p0 -> 0x0p0
0x1p0 -0x1p0 -> 0x0p0

attribute rounding-direction: roundTowardNegative

# Zeros
0x0p0 0x0p0 -> 0x0p0
-0x0p0 -0x0p0 -> -0x0p0
0x0p0 -0x0p0 -> -0x0p0
-0x0p0 0x0p0 -> -0x0p0
0x1p0 -0x1p0 -> -0x0p0

attribute rounding-direction: roundTowardZero

# Zeros
0x0p0 0x0p0 -> 0x0p0
-0x0p0 -0x0p0 -> -0x0p0
0x0p0 -0x0p0 -> 0x0p0
-0x0p0 0x0p0 -> 0x0p0
0x1p0 -0x1p0 -> 0x0p0

"""


class TestAddition(unittest.TestCase):
    def test_data(self):
        for test_case in parse_test_data(test_data):
            if test_case.actual_result != test_case.expected_result:
                self.fail(
                    "Error in test case:\n{0!r}".format(test_case)
                )
