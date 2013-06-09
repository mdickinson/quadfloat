import doctest
import unittest

import quadfloat.arithmetic
import quadfloat.binary_interchange_format


doctest_modules = [
    quadfloat.arithmetic,
    quadfloat.binary_interchange_format,
]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for module in doctest_modules:
        suite.addTests(doctest.DocTestSuite(module=module))
    return suite
