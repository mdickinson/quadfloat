import os
import unittest

from quadfloat.tests.parse_test_data import parse_test_data


class TestDataCase(unittest.TestCase):
    def runTest(self):
        with open(self.datafile) as f:
            test_data = f.read()

        for test_case in parse_test_data(test_data):
            if test_case.actual_result != test_case.expected_result:
                self.fail(
                    "Error in test case:\n{0!r}".format(test_case)
                )


TESTS_DIRECTORY = os.path.dirname(__file__)
DATA_DIRECTORY = os.path.join(TESTS_DIRECTORY, 'quadfloat_test_data')

TESTFILE_SUFFIX = '.qtest'


def find_data_files():
    for filename in os.listdir(DATA_DIRECTORY):
        if not filename.endswith(TESTFILE_SUFFIX):
            continue
        prefix = filename[:-len(TESTFILE_SUFFIX)]
        test_name = "test_{0}".format(prefix)
        path = os.path.join(DATA_DIRECTORY, filename)
        yield test_name, path


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_name, path in find_data_files():
        test_data = TestDataCase()
        test_data.datafile = path
        suite.addTest(test_data)
    return suite
