import unittest


class TestToplevel(unittest.TestCase):
    def test_import_predefined_formats(self):
        from quadfloat import binary16
        from quadfloat import binary32
        from quadfloat import binary64
        from quadfloat import binary128

    def test_import_binary_interchange_format(self):
        from quadfloat import BinaryInterchangeFormat
