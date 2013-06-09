import os
import unittest

from quadfloat.tests.parse_test_data import parse_test_data


class QuadfloatTestDataCase(unittest.TestCase):
    def eval_test_file(self, test_file):
        with open(test_file) as f:
            test_data = f.read()

        for test_case in parse_test_data(test_data):
            if test_case.actual_result != test_case.expected_result:
                self.fail(
                    "Error in test case:\n{0!r}".format(test_case)
                )


TESTS_DIRECTORY = os.path.dirname(__file__)
DATA_DIRECTORY = os.path.join(TESTS_DIRECTORY, 'quadfloat_test_data')

TESTFILE_SUFFIX = '.qtest'


def tests_for_path(path, test_name):
    def test_method(self):
        self.eval_test_file(path)
    test_method.__name__ = test_name
    return test_method


def build_test_methods():
    for filename in os.listdir(DATA_DIRECTORY):
        if not filename.endswith(TESTFILE_SUFFIX):
            continue
        prefix = filename[:-len(TESTFILE_SUFFIX)]
        test_name = "test_{0}".format(prefix)
        path = os.path.join(DATA_DIRECTORY, filename)
        test_method = tests_for_path(path, test_name)
        setattr(QuadfloatTestDataCase, test_name, test_method)


build_test_methods()
