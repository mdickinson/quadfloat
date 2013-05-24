"""
Helper class for representing a single test.

"""
from quadfloat import binary16, binary32, binary64, binary128


class ArithmeticTestCase(object):
    def __init__(self, args, result, flags, callable, attributes):
        self.args = args
        self.result = result
        self.flags = flags
        self.callable = callable
        self.attributes = attributes

    def __repr__(self):
        return "{} {} -> {} {} {}".format(
            self.callable.__name__,
            self.args,
            self.result,
            self.flags,
            self.attributes,
        )


# Mapping from operations to callables.

formats = {
    'binary16': binary16,
    'binary32': binary32,
    'binary64': binary64,
    'binary128': binary128,
}


def convertFromHexCharacter(destination):
    destination_format = formats[destination]
    return destination_format.convert_from_hex_character


operations = {
    'convertFromHexCharacter': convertFromHexCharacter,
}
