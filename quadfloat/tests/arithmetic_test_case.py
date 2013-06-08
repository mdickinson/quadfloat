"""
Helper class for representing a single test.

"""
import contextlib

from quadfloat.attributes import (
    partial_attributes,
    temporary_attributes,
)
from quadfloat.exceptions import (
    InexactException,
    OverflowException,
    UnderflowException,
)


# Context manager for catching all the exceptions that are signaled under
# default handling.

@contextlib.contextmanager
def catch_exceptions():
    signal_list = []

    def my_handler(exc):
        signal_list.append(exc)
        return exc.default_handler()

    with partial_attributes(
            invalid_operation_handler=my_handler,
            inexact_handler=my_handler,
            overflow_handler=my_handler,
            underflow_handler=my_handler,
    ):
        yield signal_list


class ArithmeticTestCase(object):
    def __init__(self, args, result, flags, operation, attributes):
        self.args = args
        self.result = result
        self.flags = flags
        self.operation = operation
        self.attributes = attributes

    def __repr__(self):
        return "{0} {1} -> {2} {3} {4}".format(
            self.operation.__name__,
            self.args,
            self.result,
            self.flags,
            self.attributes,
        )

    def execute(self):
        """
        Execute the call represented by this testcase, returning
        the results and resulting signals.

        """
        with temporary_attributes(self.attributes):
            with catch_exceptions() as exceptions:
                actual_result = self.operation(*self.args)
        actual_flags = set()
        actual_inexact = any(
            isinstance(exc, InexactException)
            for exc in exceptions)
        if actual_inexact:
            actual_flags.add('inexact')

        actual_overflow = any(
            isinstance(exc, OverflowException)
            for exc in exceptions)
        if actual_overflow:
            actual_flags.add('overflow')

        actual_underflow = any(
            isinstance(exc, UnderflowException)
            for exc in exceptions)
        if actual_underflow:
            actual_flags.add('underflow')

        return actual_result, actual_flags
