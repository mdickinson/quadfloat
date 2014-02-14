"""
Tests for BitString class.

"""
import unittest

from quadfloat.bit_string import BitString


class TestBitString(unittest.TestCase):
    def test_str_roundtrip(self):
        test_strings = [
            '',
            '0',
            '1',
            '00',
            '01',
            '10',
            '11',
            '010010101',
        ]
        for test_string in test_strings:
            bit_string = BitString(test_string)
            self.assertEqual(str(bit_string), test_string)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            BitString('123')
        with self.assertRaises(ValueError):
            BitString(' 011')
        with self.assertRaises(ValueError):
            BitString('0b011')

    def test_equality(self):
        self.assertTrue(BitString('010') == BitString('010'))
        self.assertFalse(BitString('011') == BitString('010'))
        self.assertFalse(BitString('0') == BitString('00'))
        self.assertFalse(BitString('') == BitString('0'))
        self.assertFalse(BitString('010') != BitString('010'))
        self.assertTrue(BitString('011') != BitString('010'))
        self.assertTrue(BitString('0') != BitString('00'))
        self.assertTrue(BitString('') != BitString('0'))

    def test_hashing(self):
        s = {BitString('011'), BitString('10')}
        self.assertEqual(len(s), 2)
        self.assertIn(BitString('011'), s)
        self.assertIn(BitString('10'), s)
        self.assertNotIn(BitString('010'), s)

    def test_create_from_int(self):
        self.assertEqual(
            BitString.from_int(3, 3),
            BitString('011'),
        )
