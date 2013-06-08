from quadfloat.tests.base_test_case import BaseTestCase
from quadfloat.tests.parse_test_data import parse_test_data


FAILURE_MSG_TEMPLATE = """\
Flags don't match for failed test: {0!r}
Actual flags: {1!r}
Expected flags: {2!r}
"""


test_data = """\
operation: subtraction
operation destination: binary16
operation source1: binary16
operation source2: binary16
attribute tininess-detection: afterRounding

attribute rounding-direction: roundTiesToEven

# Zeros
0x0p0 -0x0p0 -> 0x0p0
-0x0p0 0x0p0 -> -0x0p0
0x0p0 0x0p0 -> 0x0p0
-0x0p0 -0x0p0 -> 0x0p0
0x1p0 0x1p0 -> 0x0p0

# Infinities
Infinity Infinity -> NaN invalid
-Infinity -Infinity -> NaN invalid
Infinity -Infinity -> Infinity
-Infinity Infinity -> -Infinity

attribute rounding-direction: roundTiesToAway

# Zeros
0x0p0 -0x0p0 -> 0x0p0
-0x0p0 0x0p0 -> -0x0p0
0x0p0 0x0p0 -> 0x0p0
-0x0p0 -0x0p0 -> 0x0p0
0x1p0 0x1p0 -> 0x0p0

attribute rounding-direction: roundTowardPositive

# Zeros
0x0p0 -0x0p0 -> 0x0p0
-0x0p0 0x0p0 -> -0x0p0
0x0p0 0x0p0 -> 0x0p0
-0x0p0 -0x0p0 -> 0x0p0
0x1p0 0x1p0 -> 0x0p0

attribute rounding-direction: roundTowardNegative

# Zeros
0x0p0 -0x0p0 -> 0x0p0
-0x0p0 0x0p0 -> -0x0p0
0x0p0 0x0p0 -> -0x0p0
-0x0p0 -0x0p0 -> -0x0p0
0x1p0 0x1p0 -> -0x0p0

attribute rounding-direction: roundTowardZero

# Zeros
0x0p0 -0x0p0 -> 0x0p0
-0x0p0 0x0p0 -> -0x0p0
0x0p0 0x0p0 -> 0x0p0
-0x0p0 -0x0p0 -> 0x0p0
0x1p0 0x1p0 -> 0x0p0

"""


class TestSubtraction(BaseTestCase):
    def test_data(self):
        for arithmetic_test_case in parse_test_data(test_data):
            expected_result = arithmetic_test_case.result
            expected_flags = arithmetic_test_case.flags
            actual_result, actual_flags = arithmetic_test_case.execute()

            self.assertInterchangeable(
                actual_result,
                expected_result,
                msg=str(arithmetic_test_case))

            self.assertEqual(
                actual_flags,
                expected_flags,
                msg=FAILURE_MSG_TEMPLATE.format(
                    arithmetic_test_case,
                    actual_flags,
                    expected_flags,
                ),
            )
