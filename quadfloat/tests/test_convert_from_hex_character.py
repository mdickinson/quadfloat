from quadfloat import binary16
from quadfloat.tests.base_test_case import BaseTestCase


binary16_inputs = """\
0x0p0
0x0p1
NaN
-NaN
NaN(0)
NaN(1)
NaN(0001)
NaN(511)
-NaN(0)
-NaN(1)
-NaN(511)
sNaN
-sNaN
sNaN(1)
sNaN(511)
-sNaN(1)
-sNaN(511)
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
-NaN(512)
NaN(512)
sNaN(0)
-sNaN(0)
sNaN(512)
-sNaN(512)
""".splitlines()


class TestConvertFromHexCharacter(BaseTestCase):
    def test_invalid_inputs(self):
        for invalid_input in binary16_invalid_inputs:
            with self.assertRaises(ValueError):
                binary16.convert_from_hex_character(invalid_input)

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
