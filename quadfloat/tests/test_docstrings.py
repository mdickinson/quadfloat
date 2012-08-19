import doctest

import quadfloat.arithmetic
import quadfloat.binary_interchange_format


if __name__ == '__main__':
    doctest.testmod(quadfloat.arithmetic)
    doctest.testmod(quadfloat.binary_interchange_format)
