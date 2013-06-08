import unittest

from quadfloat.parsing import (
    parse_finite_decimal,
    parse_finite_hexadecimal,
    parse_infinity,
    parse_nan,
)
from quadfloat.tests.base_test_case import BaseTestCase


class TestParsing(BaseTestCase):
    def test_parse_finite_decimal(self):
        self.assertEqual(parse_finite_decimal('0'), (False, 0, 0))
        self.assertEqual(parse_finite_decimal('-0'), (True, 0, 0))
        self.assertEqual(parse_finite_decimal('23'), (False, 0, 23))
        self.assertEqual(parse_finite_decimal('23.'), (False, 0, 23))
        self.assertEqual(parse_finite_decimal('0.7'), (False, -1, 7))
        self.assertEqual(parse_finite_decimal('.7'), (False, -1, 7))
        self.assertEqual(parse_finite_decimal('2.3'), (False, -1, 23))
        self.assertEqual(parse_finite_decimal('-2.3'), (True, -1, 23))
        self.assertEqual(parse_finite_decimal('+2.3'), (False, -1, 23))
        self.assertEqual(parse_finite_decimal('2.30'), (False, -2, 230))
        self.assertEqual(parse_finite_decimal('1e0'), (False, 0, 1))
        self.assertEqual(parse_finite_decimal('1e1'), (False, 1, 1))
        self.assertEqual(parse_finite_decimal('1e+1'), (False, 1, 1))
        self.assertEqual(parse_finite_decimal('1e-1'), (False, -1, 1))
        self.assertEqual(parse_finite_decimal('1E2'), (False, 2, 1))
        self.assertEqual(parse_finite_decimal('123456'), (False, 0, 123456))
        self.assertEqual(parse_finite_decimal('12345.6e1'), (False, 0, 123456))
        self.assertEqual(parse_finite_decimal('1234.56e2'), (False, 0, 123456))
        self.assertEqual(parse_finite_decimal('123.456e3'), (False, 0, 123456))
        self.assertEqual(parse_finite_decimal('12.3456e4'), (False, 0, 123456))
        self.assertEqual(parse_finite_decimal('1.23456e5'), (False, 0, 123456))
        self.assertEqual(parse_finite_decimal('.123456e6'), (False, 0, 123456))
        self.assertEqual(
            parse_finite_decimal('.0123456e7'),
            (False, 0, 123456),
        )
        self.assertEqual(
            parse_finite_decimal('0.0123456e7'),
            (False, 0, 123456),
        )

        bad_inputs = [
            '0p0',
            '123.0.',
            '123..1',
            '123,4',
            '10e',
            '++1',
            '--1',
            '+-1',
            '-+1',
        ]
        for input in bad_inputs:
            with self.assertRaises(ValueError):
                parse_finite_decimal(input)

    def test_parse_finite_hexadecimal(self):
        self.assertEqual(
            parse_finite_hexadecimal('0x3p+23'),
            (False, 23, 3),
        )
        self.assertEqual(
            parse_finite_hexadecimal('0x3p0'),
            (False, 0, 3),
        )

        bad_inputs = [
            '0p0',  # no hexadecimal prefix
            '0x0',  # no exponent
            '0x0p',
            '0x0p1q',  # invalid postfix
        ]
        for input in bad_inputs:
            with self.assertRaises(ValueError):
                parse_finite_hexadecimal(input)

    def test_parse_infinity(self):
        self.assertEqual(parse_infinity('inf'), False)
        self.assertEqual(parse_infinity('+inf'), False)
        self.assertEqual(parse_infinity('-inf'), True)
        self.assertEqual(parse_infinity('infinity'), False)
        self.assertEqual(parse_infinity('+infinity'), False)
        self.assertEqual(parse_infinity('-infinity'), True)
        self.assertEqual(parse_infinity('Infinity'), False)
        self.assertEqual(parse_infinity('-INF'), True)

        bad_inputs = [
            'infinity2',
            ' inf',
            'INFIINITY',
            'i',
            'in',
            'infi',
            'infin',
            'infini',
            'infinit',
            '0',
        ]
        for input in bad_inputs:
            with self.assertRaises(ValueError):
                parse_infinity(input)

    def test_parse_nan(self):
        self.assertEqual(parse_nan('nan'), (False, False, 0))
        self.assertEqual(parse_nan('+nan'), (False, False, 0))
        self.assertEqual(parse_nan('-nan'), (True, False, 0))
        self.assertEqual(parse_nan('snan'), (False, True, 0))
        self.assertEqual(parse_nan('-snan'), (True, True, 0))
        self.assertEqual(parse_nan('nan(123)'), (False, False, 123))
        self.assertEqual(parse_nan('NaN'), (False, False, 0))
        self.assertEqual(parse_nan('Nan'), (False, False, 0))
        self.assertEqual(parse_nan('NAN'), (False, False, 0))
        self.assertEqual(parse_nan('snan'), (False, True, 0))
        self.assertEqual(parse_nan('sNaN'), (False, True, 0))
        self.assertEqual(parse_nan('sNan'), (False, True, 0))
        self.assertEqual(parse_nan('sNAN'), (False, True, 0))
        self.assertEqual(parse_nan('Snan'), (False, True, 0))
        self.assertEqual(parse_nan('SNaN'), (False, True, 0))
        self.assertEqual(parse_nan('SNan'), (False, True, 0))
        self.assertEqual(parse_nan('SNAN'), (False, True, 0))

        bad_inputs = [
            'nan0',
            'nan()',
            'nan(-123)',
        ]
        for input in bad_inputs:
            with self.assertRaises(ValueError):
                parse_infinity(input)
