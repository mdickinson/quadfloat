import unittest

from quadfloat import binary16
from quadfloat.tests.base_test_case import BaseTestCase
from quadfloat.tests.arithmetic_test_case import parse_test_data


binary16_inputs = """\
0x0p0
0x0p1
""".splitlines()


binary16_invalid_inputs = """\
0
0x0
0p0
++0x0p0
-+0x0p0
+-0x0p0
--0x0p0
+ 0x0p0
i
in
infi
infin
infini
infinit
infinityy
""".splitlines()


# 16-bit test values, round-half-to-even
test16 = """\
operation: convertFromHexCharacter
operation destination: binary16

attribute rounding-direction: round-ties-to-even
attribute tininess-detection: after-rounding

# Tiny numbers.
0x0p-26 -> 0x0p-24
0x0p-1000 -> 0x0p-24
0x0.8p-26 -> 0x0p-24 inexact underflow
0x1p-26 -> 0x0p-24 inexact underflow
0x1.8p-26 -> 0x0p-24 inexact underflow
0x2p-26 -> 0x0p-24 inexact underflow
0x2.0000000000000000001p-26 -> 0x1p-24 inexact underflow
0x2.8p-26 -> 0x1p-24 inexact underflow
0x3p-26 -> 0x1p-24 inexact underflow
0x3.8p-26 -> 0x1p-24 inexact underflow
0x4p-26 -> 0x1p-24 underflow
0x4.8p-26 -> 0x1p-24 inexact underflow
0x5p-26 -> 0x1p-24 inexact underflow
0x5.8p-26 -> 0x1p-24 inexact underflow
0x5.fffffffffffffffffffp-26 -> 0x1p-24 inexact underflow
0x6p-26 -> 0x2p-24 inexact underflow
0x6.8p-26 -> 0x2p-24 inexact underflow
0x7p-26 -> 0x2p-24 inexact underflow
0x7.8p-26 -> 0x2p-24 inexact underflow
0x8p-26 -> 0x2p-24 underflow
0x9p-26 -> 0x2p-24 inexact underflow
0xap-26 -> 0x2p-24 inexact underflow
0xbp-26 -> 0x3p-24 inexact underflow

# Subnormal / normal boundary.
0x0.ffcp-14 -> 0x0.ffcp-14 underflow
0x0.ffep-14 -> 0x1p-14 inexact underflow
0x0.ffefffffffp-14 -> 0x1p-14 inexact underflow
0x0.fffp-14 -> 0x1p-14 inexact
0x0.ffffffffffp-14 -> 0x1p-14 inexact
0x1p-14 -> 0x1p-14
0x1.001p-14 -> 0x1p-14 inexact
0x1.002p-14 -> 0x1p-14 inexact
0x1.00200000001p-14 -> 0x1.004p-14 inexact
0x1.003p-14 -> 0x1.004p-14 inexact
0x1.004p-14 -> 0x1.004p-14
0x1.005p-14 -> 0x1.004p-14 inexact
0x1.005fffffffp-14 -> 0x1.004p-14 inexact
0x1.006p-14 -> 0x1.008p-14 inexact
0x1.007p-14 -> 0x1.008p-14 inexact
0x1.008p-14 -> 0x1.008p-14

# Testing round-half-to-even, numbers near 1.
0x0.ffap0 -> 0x0.ffap0
0x0.ffa8p0 -> 0x0.ffap0 inexact
0x0.ffbp0 -> 0x0.ffcp0 inexact
0x0.ffb8p0 -> 0x0.ffcp0 inexact
0x0.ffcp0 -> 0x0.ffcp0
0x0.ffc8p0 -> 0x0.ffcp0 inexact
0x0.ffdp0 -> 0x0.ffcp0 inexact
0x0.ffd8p0 -> 0x0.ffep0 inexact
0x0.ffep0 -> 0x0.ffep0
0x0.ffe8p0 -> 0x0.ffep0 inexact
0x0.fffp0 -> 0x1p0 inexact
0x0.fff8p0 -> 0x1p0 inexact
0x0.ffffffffp0 -> 0x1p0 inexact
0x1p0 -> 0x1p0
0x1.001p0 -> 0x1p0 inexact
0x1.002p0 -> 0x1p0 inexact
0x1.00200000001p0 -> 0x1.004p0 inexact
0x1.003p0 -> 0x1.004p0 inexact
0x1.004p0 -> 0x1.004p0
0x1.005p0 -> 0x1.004p0 inexact
0x1.005fffffffp0 -> 0x1.004p0 inexact
0x1.006p0 -> 0x1.008p0 inexact
0x1.007p0 -> 0x1.008p0 inexact
0x1.008p0 -> 0x1.008p0

# Integers
0x1p0 -> 0x1p0
0x2p0 -> 0x2p0
0x7ffp0 -> 0x7ffp0
0x800p0 -> 0x800p0
0x801p0 -> 0x800p0 inexact
0x802p0 -> 0x802p0
0x803p0 -> 0x804p0 inexact

# Testing near overflow boundary.
0x0.ffcp16 -> 0x0.ffcp16
0x0.ffdp16 -> 0x0.ffcp16 inexact
0x0.ffep16 -> 0x0.ffep16
0x0.ffe8p16 -> 0x0.ffep16 inexact
0x0.ffefffffffffp16 -> 0x0.ffep16 inexact
0x0.fffp16 -> Infinity inexact overflow
0x0.fff8p16 -> Infinity inexact overflow
0x1p16 -> Infinity inexact overflow
0x1.0008p16 -> Infinity inexact overflow
0x1.001p16 -> Infinity inexact overflow
0x1.0018p16 -> Infinity inexact overflow
0x1.002p16 -> Infinity inexact overflow

# Infinities
inf -> Infinity
infinity -> Infinity
Infinity -> Infinity
iNFinItY -> Infinity
INF -> Infinity
-inf -> -Infinity
-infinity -> -Infinity
-Infinity -> -Infinity
-iNFinItY -> -Infinity
-INF -> -Infinity

# Tests for underflow before rounding.
attribute tininess-detection: before-rounding
0x0.ffcp-14 -> 0x0.ffcp-14 underflow
0x0.ffep-14 -> 0x1p-14 inexact underflow
0x0.ffefffffffp-14 -> 0x1p-14 inexact underflow
0x0.fffp-14 -> 0x1p-14 inexact underflow
0x0.ffffffffffp-14 -> 0x1p-14 inexact underflow
0x1p-14 -> 0x1p-14
0x1.001p-14 -> 0x1p-14 inexact
0x1.002p-14 -> 0x1p-14 inexact
0x1.00200000001p-14 -> 0x1.004p-14 inexact
0x1.003p-14 -> 0x1.004p-14 inexact
0x1.004p-14 -> 0x1.004p-14

# Now some of the same tests with round-ties-to-away.
attribute rounding-direction: round-ties-to-away
attribute tininess-detection: after-rounding

# Tiny numbers.
0x0p-26 -> 0x0p-24
0x0p-1000 -> 0x0p-24
0x0.8p-26 -> 0x0p-24 inexact underflow
0x1p-26 -> 0x0p-24 inexact underflow
0x1.8p-26 -> 0x0p-24 inexact underflow
0x2p-26 -> 0x1p-24 inexact underflow
0x2.0000000000000000001p-26 -> 0x1p-24 inexact underflow
0x2.8p-26 -> 0x1p-24 inexact underflow
0x3p-26 -> 0x1p-24 inexact underflow
0x3.8p-26 -> 0x1p-24 inexact underflow
0x4p-26 -> 0x1p-24 underflow
0x4.8p-26 -> 0x1p-24 inexact underflow
0x5p-26 -> 0x1p-24 inexact underflow
0x5.8p-26 -> 0x1p-24 inexact underflow
0x5.fffffffffffffffffffp-26 -> 0x1p-24 inexact underflow
0x6p-26 -> 0x2p-24 inexact underflow
0x6.8p-26 -> 0x2p-24 inexact underflow
0x7p-26 -> 0x2p-24 inexact underflow
0x7.8p-26 -> 0x2p-24 inexact underflow
0x8p-26 -> 0x2p-24 underflow
0x9p-26 -> 0x2p-24 inexact underflow
0xap-26 -> 0x3p-24 inexact underflow
0xbp-26 -> 0x3p-24 inexact underflow

"""


class TestConvertFromHexCharacter(BaseTestCase):
    def test_16(self):
        for arithmetic_test_case in parse_test_data(test16):
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
                msg="""\
Flags don't match for failed test: {0!r}
Actual flags: {1!r}
Expected flags: {2!r}
""".format(arithmetic_test_case, actual_flags, expected_flags)
            )

    def test_invalid_inputs(self):
        for input in binary16_invalid_inputs:
            with self.assertRaises(ValueError):
                binary16.convert_from_hex_character(input)

    def test_validity(self):
        for input in binary16_inputs:
            result = binary16.convert_from_hex_character(input)
            self.assertEqual(result.format, binary16)

    def test_against_convert_from_int(self):
        self.assertInterchangeable(
            binary16.convert_from_hex_character('0x0p0'),
            binary16.convert_from_int(0),
        )
        self.assertInterchangeable(
            binary16.convert_from_hex_character('0x1p0'),
            binary16.convert_from_int(1),
        )
        self.assertInterchangeable(
            binary16.convert_from_hex_character('-0x1p0'),
            binary16.convert_from_int(-1),
        )
        self.assertInterchangeable(
            binary16.convert_from_hex_character('0xap0'),
            binary16.convert_from_int(10),
        )
        self.assertInterchangeable(
            binary16.convert_from_hex_character('0xap1'),
            binary16.convert_from_int(20),
        )
        self.assertInterchangeable(
            binary16.convert_from_hex_character('0xap-1'),
            binary16.convert_from_int(5),
        )
        self.assertInterchangeable(
            binary16.convert_from_hex_character('0x1p10'),
            binary16.convert_from_int(1024),
        )

    def test_against_convert_from_float(self):
        self.assertInterchangeable(
            binary16.convert_from_hex_character('0x1.8p0'),
            binary16(1.5),
        )


if __name__ == '__main__':
    unittest.main()
