import unittest


class TestToplevel(unittest.TestCase):
    def test_import_predefined_formats(self):
        from quadfloat.api import binary16
        from quadfloat.api import binary32
        from quadfloat.api import binary64
        from quadfloat.api import binary128
        from quadfloat.api import BinaryInterchangeFormat
        self.assertIsInstance(binary16, BinaryInterchangeFormat)
        self.assertIsInstance(binary32, BinaryInterchangeFormat)
        self.assertIsInstance(binary64, BinaryInterchangeFormat)
        self.assertIsInstance(binary128, BinaryInterchangeFormat)
