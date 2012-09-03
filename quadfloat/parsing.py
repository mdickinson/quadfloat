"""
Support for parsing strings representing decimal or hexadecimal numbers.

"""
import re

# Regular expressions for numeric sequences.

sign = "[-+]"

period = "\."

decimal_digit = "[0123456789]"

hexadecimal_digit = "[0123456789aAbBcCdDeEfF]"

optional_sign = "(?P<sign>{sign})?".format(sign=sign)

significand_template = """\
(?={digit}|{period}{digit})         # lookhead ensuring at least one digit
(?P<integral>{digit}*)              # possibly empty integer part
(?:{period}(?P<fraction>{digit}*))? # possibly empty fractional part
"""

decimal_significand = significand_template.format(
    digit=decimal_digit,
    period=period,
)

hexadecimal_significand = significand_template.format(
    digit=hexadecimal_digit,
    period=period,
)

decimal_exponent = "(?P<exponent>{sign}?{decimal_digit}+)".format(
    sign=sign,
    decimal_digit=decimal_digit,
)

decimal_number = """\
{optional_sign}
{decimal_significand}
(?:[eE]{decimal_exponent})?
""".format(
    optional_sign=optional_sign,
    decimal_significand=decimal_significand,
    decimal_exponent=decimal_exponent,
)

hexadecimal_number = """\
{optional_sign}
0[xX]{hexadecimal_significand}
[pP]{decimal_exponent}
""".format(
    optional_sign=optional_sign,
    hexadecimal_significand=hexadecimal_significand,
    decimal_exponent=decimal_exponent,
)

infinity = """\
{optional_sign}
[iI][nN][fF](?:[iI][nN][iI][tT][yY])?
""".format(
    optional_sign=optional_sign,
)

payload = "\((?P<payload>{decimal_digit}+)\)".format(
    decimal_digit=decimal_digit
)

nan = """\
{optional_sign}
(?P<signaling>[sS])?
[nN][aA][nN]
(?:{payload})?
""".format(
    optional_sign=optional_sign,
    payload=payload,
)


def parse_finite_decimal(s):
    """
    Given a string representing a decimal number,
    return a triple (sign, exponent, significand).

    """
    match = re.match(decimal_number, s, re.VERBOSE)
    if match is None or match.group() != s:
        raise ValueError("invalid numeric string")

    sign = match.group('sign') == '-'
    integral = match.group('integral')
    fraction = match.group('fraction') or ''
    exponent = match.group('exponent') or '0'
    return (
        sign,
        int(exponent) - len(fraction),
        int(integral + fraction)
    )


def parse_finite_hexadecimal(s):
    """
    Given a string representing a hexadecimal number,
    return a triple (sign, exponent, significand).

    """
    match = re.match(hexadecimal_number, s, re.VERBOSE)
    if match is None or match.group() != s:
        raise ValueError("invalid numeric string")

    sign = match.group('sign') == '-'
    integral = match.group('integral')
    fraction = match.group('fraction') or ''
    exponent = match.group('exponent')
    return (
        sign,
        int(exponent) - 4 * len(fraction),
        int(integral + fraction, 16)
    )


def parse_infinity(s):
    """
    Given a string representing an infinity,
    return a single boolean giving its sign.

    """
    match = re.match(infinity, s, re.VERBOSE)
    if match is None or match.group() != s:
        raise ValueError("invalid numeric string")

    sign = match.group('sign') == '-'
    return sign


def parse_nan(s):
    """
    Given a string representing a nan,
    return a triple (sign, signaling, payload).

    """
    match = re.match(nan, s, re.VERBOSE)
    if match is None or match.group() != s:
        raise ValueError("invalid numeric string")

    sign = match.group('sign') == '-'
    signaling = match.group('signaling') is not None
    payload = int(match.group('payload') or '0')
    return sign, signaling, payload
