"""
Helper class for representing a single test.

"""
from quadfloat.attributes import (
    partial_attributes,
    temporary_attributes,
)
import quadfloat.binary_interchange_format


def identifying_string(binary_float):
    """
    String giving a faithful, unambiguous representation of this binary float
    and its format.  Two binary floats are interchangeable for all practical
    purposes if their identifying strings are identical.

    """
    # XXX We actually use this for all result types;  fix accordingly.
    # (TestCase could also check return type.)
    if isinstance(binary_float, int):
        return str(binary_float)

    fmt = binary_float.format
    return "{0} (format {1})".format(
        fmt.convert_to_hex_character(binary_float),
        binary_float.format,
    )


class SignalCollector(object):
    def __init__(self):
        self.flags = set()
        self.set_handlers = partial_attributes(
            inexact_handler=self.set_flag('inexact'),
            invalid_operation_handler=self.set_flag('invalid'),
            overflow_handler=self.set_flag('overflow'),
            underflow_handler=self.set_flag('underflow'),
            divide_by_zero_handler=self.set_flag('zero-division')
        )

    def set_flag(self, flag):
        def exception_handler(exception, attributes):
            self.flags.add(flag)
            return exception.default_handler(attributes)
        return exception_handler

    def __enter__(self):
        self.set_handlers.__enter__()
        return self.flags

    def __exit__(self, *exc_info):
        return self.set_handlers.__exit__(*exc_info)


class ArithmeticOperation(object):
    """
    Abstract base class.

    """
    def __str__(self):
        raise NotImplementedError

    def __call__(self, *args):
        # An arithmetic operation should be callable.
        raise NotImplementedError


class HomogeneousOperation(ArithmeticOperation):
    def __init__(self, method_name):
        self.method_name = method_name

    def __str__(self):
        return self.method_name

    def __call__(self, *args):
        method = getattr(quadfloat.binary_interchange_format, self.method_name)
        return method(*args)


class FormatOfOperation(ArithmeticOperation):
    def __init__(self, format, method_name):
        self.format = format
        self.method_name = method_name

    def __str__(self):
        return "{}-{}".format(self.format, self.method_name)

    def __call__(self, *args):
        method = getattr(self.format, self.method_name)
        return method(*args)


class ArithmeticTestResult(object):
    def __init__(self, result, flags):
        self.result = result
        self.flags = flags

    def __repr__(self):
        return "{0} {1}".format(
            identifying_string(self.result),
            ' '.join(sorted(self.flags)),
        )

    def __eq__(self, other):
        self_result = identifying_string(self.result)
        other_result = identifying_string(other.result)

        return (
            self_result == other_result and
            self.flags == other.flags
        )

    def __ne__(self, other):
        return not self == other


class ArithmeticTestCase(object):
    def __init__(self, attributes, operation, operands, expected_result,
                 source_file, line_number):
        self.attributes = attributes
        self.operation = operation
        self.operands = operands
        self.expected_result = expected_result
        self.source_file = source_file
        self.line_number = line_number
        self._actual_result = None

    @property
    def actual_result(self):
        if self._actual_result is None:
            self._actual_result = self._execute()
        return self._actual_result

    def _execute(self):
        """
        Execute the call represented by this testcase, returning
        the results and resulting signals.

        """
        with temporary_attributes(self.attributes):
            with SignalCollector() as actual_flags:
                actual_result = self.operation(*self.operands)
        return ArithmeticTestResult(
            result=actual_result,
            flags=actual_flags,
        )

    def __repr__(self):
        return """\
File {self.source_file!r}, line {self.line_number}
Attributes: {self.attributes}
Operation: {self.operation}
Operands: {self.operands}
Expected result = {self.expected_result}
Actual result   = {self.actual_result}
        """.format(self=self)
