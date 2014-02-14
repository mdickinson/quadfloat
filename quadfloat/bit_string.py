import re
import sys


class BitString(object):
    """
    Class representing a bit string.

    """
    @classmethod
    def _raw(cls, width, value_as_int):
        """
        Private constructor for direct construction of a BitString.

        """
        # A bitstring is represented internally as a combination
        # of an int and a width.
        self = object.__new__(cls)
        if not width >= 0:
            raise ValueError(
                "Invalid width {0}.  Width should be "
                "positive.".format(width)
            )
        if not 0 <= value_as_int < (1 << width):
            raise ValueError(
                "Invalid value_as_int {0}. value_as_int should satisfy "
                "0 <= value_as_int < 2**width.".format(value_as_int))
        self.value_as_int = int(value_as_int)
        self.width = int(width)
        return self

    @classmethod
    def from_int(cls, width, value_as_int):
        return cls._raw(width=width, value_as_int=value_as_int)

    def __new__(cls, s):
        if not re.match(r'[01]*\Z', s):
            raise ValueError("Invalid bit string.")
        value_as_int = 0 if s == '' else int(s, base=2)
        return cls._raw(width=len(s), value_as_int=value_as_int)

    def __str__(self):
        if self.width:
            return "{:0{}b}".format(self.value_as_int, self.width)
        else:
            return ""

    def __repr__(self):
        return "BitString({!r})".format(str(self))

    def __eq__(self, other):
        return (
            self.width == other.width and
            self.value_as_int == other.value_as_int
        )

    def __hash__(self):
        return hash(('BitString', self.width, self.value_as_int))

    if sys.version_info[0] == 2:
        # != is automatically inferred from == for Python 3.
        def __ne__(self, other):
            return not (self == other)
