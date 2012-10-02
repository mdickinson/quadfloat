from quadfloat.binary_interchange_format import BinaryInterchangeFormat

__all__ = [
    'BinaryInterchangeFormat',
    'binary16',
    'binary32',
    'binary64',
    'binary128',
]

binary16 = BinaryInterchangeFormat(width=16)
binary32 = BinaryInterchangeFormat(width=32)
binary64 = BinaryInterchangeFormat(width=64)
binary128 = BinaryInterchangeFormat(width=128)
