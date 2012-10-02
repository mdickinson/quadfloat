import unittest

from quadfloat import binary16


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
""".splitlines()


class TestConvertFromHexCharacter(unittest.TestCase):
    def test_invalid_inputs(self):
        for input in binary16_invalid_inputs:
            with self.assertRaises(ValueError):
                binary16.convert_from_hex_character(input)

    def test_validity(self):
        for input in binary16_inputs:
            result = binary16.convert_from_hex_character(input)
            self.assertEqual(result.format, binary16)

    def test_against_convert_from_int(self):
        self.assertEqual(
            binary16.convert_from_hex_character('0x0p0'),
            binary16.convert_from_int(0),
        )
        self.assertEqual(
            binary16.convert_from_hex_character('0x1p0'),
            binary16.convert_from_int(1),
        )
        self.assertEqual(
            binary16.convert_from_hex_character('-0x1p0'),
            binary16.convert_from_int(-1),
        )
        self.assertEqual(
            binary16.convert_from_hex_character('0xap0'),
            binary16.convert_from_int(10),
        )
        self.assertEqual(
            binary16.convert_from_hex_character('0xap1'),
            binary16.convert_from_int(20),
        )
        self.assertEqual(
            binary16.convert_from_hex_character('0xap-1'),
            binary16.convert_from_int(5),
        )

    def test_against_convert_from_float(self):
        self.assertEqual(
            binary16.convert_from_hex_character('0x1.8p0'),
            binary16(1.5),
        )


if __name__ == '__main__':
    unittest.main()
