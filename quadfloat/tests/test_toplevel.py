class TestToplevel(unittest.TestCase):
    def test_import_predefined_formats(self):
        from quadfloat import float16
        from quadfloat import float32
        from quadfloat import float64
        from quadfloat import float128

    def test_import_binary_interchange_format(self):
        from quadfloat import BinaryInterchangeFormat
        
