"""
Python 2 / 3 compatibility code.

"""
import sys

if sys.version_info[0] == 2:
    from future_builtins import map as _map
    from future_builtins import zip as _zip
    STRING_TYPES = str, unicode
    INTEGER_TYPES = int, long

    # Values used to compute hashes.
    if sys.maxsize == 2**31 - 1:
        _PyHASH_MODULUS = 2**31 - 1
    elif sys.maxsize == 2**63 - 1:
        _PyHASH_MODULUS = 2**61 - 1
    else:
        raise ValueError("Unexpected value for sys.maxsize.")
    _PyHASH_2INV = pow(2, _PyHASH_MODULUS - 2, _PyHASH_MODULUS)
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
            hex_n = '{0:x}'.format(n)
            return 4*len(hex_n) - correction[hex_n[0]]
    else:
        def bit_length(n):
            return n.bit_length()

    import __builtin__ as builtins

else:
    _map = map
    _zip = zip
    STRING_TYPES = str,
    INTEGER_TYPES = int,

    _PyHASH_MODULUS = sys.hash_info.modulus
    _PyHASH_2INV = pow(2, _PyHASH_MODULUS - 2, _PyHASH_MODULUS)
    _PyHASH_INF = sys.hash_info.inf
    _PyHASH_NINF = -sys.hash_info.inf
    _PyHASH_NAN = sys.hash_info.nan

    bit_length = int.bit_length
    import builtins
