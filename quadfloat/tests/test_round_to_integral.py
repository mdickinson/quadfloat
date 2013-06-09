from quadfloat.tests.base_test_case import BaseTestCase
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


class TestRoundToIntegral(BaseTestCase):
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
