import doctest
import unittest

import quadfloat.arithmetic
import quadfloat.binary_interchange_format


def load_tests(loader, tests, ignore):
    modules = [
        quadfloat.arithmetic,
        quadfloat.binary_interchange_format,
    ]
    for module in modules:
        tests.addTests(doctest.DocTestSuite(module=module))
    return tests


if __name__ == '__main__':
    unittest.main()
