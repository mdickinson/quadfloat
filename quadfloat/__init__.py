from quadfloat.binary_interchange_format import BinaryInterchangeFormat

__all__ = [
    'BinaryInterchangeFormat',
    'float16',
    'float32',
    'float64',
    'float128',
]

float16 = BinaryInterchangeFormat(width=16)
float32 = BinaryInterchangeFormat(width=32)
float64 = BinaryInterchangeFormat(width=64)
float128 = BinaryInterchangeFormat(width=128)
