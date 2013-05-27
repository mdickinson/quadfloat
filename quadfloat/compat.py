"""
Python 2 / 3 compatibility code.

"""
import sys

if sys.version_info[0] == 2:
    from future_builtins import map as _map
    from future_builtins import zip as _zip
    STRING_TYPES = str, unicode
    INTEGER_TYPES = int, long

    import binascii as _binascii

    def _int_to_bytes(n, length):
        return _binascii.unhexlify(format(n, '0{0}x'.format(2 * length)))[::-1]

    def _int_from_bytes(bs):
        return int(_binascii.hexlify(bs[::-1]), 16)

    def _bytes_from_iterable(ns):
        """
        Create a bytestring from an iterable of integers.

        Each element of the iterable should be in range(256).

        """
        return ''.join(chr(n) for n in ns)

    # Values used to compute hashes.
    _PyHASH_MODULUS = None
    _PyHASH_2INV = None
    _PyHASH_INF = hash(float('inf'))
    _PyHASH_NINF = hash(float('-inf'))
    _PyHASH_NAN = hash(float('nan'))

    try:
        int.bit_length
    except AttributeError:
        def bit_length(n, correction={
                '0': 4, '1': 3, '2': 2, '3': 2,
                '4': 1, '5': 1, '6': 1, '7': 1,
                '8': 0, '9': 0, 'a': 0, 'b': 0,
                'c': 0, 'd': 0, 'e': 0, 'f': 0}):
            """
            Number of bits in binary representation of the positive integer n,
            or 0 if n == 0.

            """
            if n < 0:
                raise ValueError(
                    "The argument to _nbits should be nonnegative.")
            hex_n = "%x" % n
            return 4*len(hex_n) - correction[hex_n[0]]
    else:
        def bit_length(n):
            return n.bit_length()


else:
    _map = map
    _zip = zip
    STRING_TYPES = str,
    INTEGER_TYPES = int,

    _int_to_bytes = lambda n, length: n.to_bytes(length, byteorder='little')
    _int_from_bytes = lambda bs: int.from_bytes(bs, byteorder='little')
    _bytes_from_iterable = bytes

    _PyHASH_MODULUS = sys.hash_info.modulus
    _PyHASH_2INV = pow(2, _PyHASH_MODULUS - 2, _PyHASH_MODULUS)
    _PyHASH_INF = sys.hash_info.inf
    _PyHASH_NINF = -sys.hash_info.inf
    _PyHASH_NAN = sys.hash_info.nan

    bit_length = int.bit_length
