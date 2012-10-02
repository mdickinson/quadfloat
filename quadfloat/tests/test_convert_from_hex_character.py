import unittest

from quadfloat import float16


float16_inputs = """\
0x0p0
0x0p1
""".splitlines()


float16_invalid_inputs = """\
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
        for input in float16_invalid_inputs:
            with self.assertRaises(ValueError):
                float16.convert_from_hex_character(input)

    def test_validity(self):
        for input in float16_inputs:
            result = float16.convert_from_hex_character(input)
            self.assertEqual(result.format, float16)

    def test_against_convert_from_int(self):
        self.assertEqual(
            float16.convert_from_hex_character('0x0p0'),
            float16.convert_from_int(0),
        )
        self.assertEqual(
            float16.convert_from_hex_character('0x1p0'),
            float16.convert_from_int(1),
        )
        self.assertEqual(
            float16.convert_from_hex_character('0xap0'),
            float16.convert_from_int(10),
        )

    def test_against_convert_from_float(self):
        self.assertEqual(
            float16.convert_from_hex_character('0x1.8p0'),
            float16(1.5),
        )


if __name__ == '__main__':
    unittest.main()
