"""
Helper class for representing a single test.

"""
import contextlib

from quadfloat.attributes import (
    partial_attributes,
    temporary_attributes,
)


class SignalCollector(object):
    def __init__(self):
        self.flags = set()

    def set_flag(self, flag):
        def exception_handler(exc):
            self.flags.add(flag)
            return exc.default_handler()
        return exception_handler


@contextlib.contextmanager
def catch_exceptions():
    """
    Context manager for catching all the exceptions that are signaled under
    default handling.

    """
    collector = SignalCollector()

    with partial_attributes(
            inexact_handler=collector.set_flag('inexact'),
            invalid_operation_handler=collector.set_flag('invalid'),
            overflow_handler=collector.set_flag('overflow'),
            underflow_handler=collector.set_flag('underflow'),
    ):
        yield collector.flags


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
            with catch_exceptions() as actual_flags:
                actual_result = self.operation(*self.args)
        return actual_result, actual_flags
