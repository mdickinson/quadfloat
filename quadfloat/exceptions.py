"""
Standard exceptions.

"""
from quadfloat.status_flags import (
    divideByZero,
    inexact,
    invalid,
    overflow,
    underflow,
)


class InvalidOperationException(object):
    """
    Class representing an InvalidOperation exception.

    """
    def __init__(self, format):
        self.format = format

    def default_handler(self, attributes):
        attributes.flag_set.add(invalid)
        return self.format._nan(
            sign=False,
            signaling=False,
            payload=0,
        )

    def signal(self, attributes):
        return attributes.invalid_operation_handler(self, attributes)


class InvalidIntegerOperationException(object):
    """
    Class representing an invalid operation that returns an integer.

    """
    def __init__(self, payload):
        # 'payload' is the value to return in non-stop mode.
        self.payload = payload

    def default_handler(self, attributes):
        attributes.flag_set.add(invalid)
        return self.payload

    def signal(self, attributes):
        return attributes.invalid_operation_handler(self, attributes)


class InvalidBooleanOperationException(object):
    def __init__(self):
        pass

    def default_handler(self, attributes):
        attributes.flag_set.add(invalid)
        raise ValueError("Invalid operation returning a boolean.")

    def signal(self, attributes):
        return attributes.invalid_operation_handler(self, attributes)


class InvalidInvalidOperationException(object):
    """
    Class representing an invalid operation that has no meaningful return
    value.

    """
    def default_handler(self, attributes):
        attributes.flag_set.add(invalid)
        raise ValueError("Invalid operation returning an integer.")

    def signal(self, attributes):
        return attributes.invalid_operation_handler(self, attributes)


class SignalingNaNException(object):
    """
    InvalidOperation exception signaled as a result of an arithmetic
    operation encountering a signaling NaN.

    """
    def __init__(self, snan):
        # Default result.  (XXX actually a quiet NaN, not a signaling
        # one.)
        self.snan = snan

    def default_handler(self, attributes):
        attributes.flag_set.add(invalid)
        return self.snan

    def signal(self, attributes):
        return attributes.invalid_operation_handler(self, attributes)


class InexactException(object):
    """
    Exception signaled when a result is inexact.

    """
    def __init__(self, rounded):
        self.rounded = rounded

    def default_handler(self, attributes):
        attributes.flag_set.add(inexact)
        return self.rounded

    def signal(self, attributes):
        return attributes.inexact_handler(self, attributes)


class UnderflowException(object):
    """
    Exception signaled when a result is non-zero and tiny.

    """
    def __init__(self, rounded, inexact):
        self.rounded = rounded
        self.inexact = inexact

    def default_handler(self, attributes):
        if self.inexact:
            attributes.flag_set.add(underflow)
            inexact_exception = InexactException(self.rounded)
            return inexact_exception.signal(attributes)
        else:
            # underflow flag *not* raised, as per section 7.5
            return self.rounded

    def signal(self, attributes):
        return attributes.underflow_handler(self, attributes)


class OverflowException(object):
    """
    Exception signaled when a result overflows.

    """
    def __init__(self, rounded):
        self.rounded = rounded

    def default_handler(self, attributes):
        attributes.flag_set.add(overflow)
        inexact_exception = InexactException(self.rounded)
        return inexact_exception.signal(attributes)

    def signal(self, attributes):
        return attributes.overflow_handler(self, attributes)


class DivideByZeroException(object):
    """
    Exception signaled when an operation with finite operands
    results in an exact infinity.

    """
    def __init__(self, sign, format):
        self.sign = sign
        self.format = format

    def default_handler(self, attributes):
        attributes.flag_set.add(divideByZero)
        return self.format._infinite(self.sign)

    def signal(self, attributes):
        return attributes.divide_by_zero_handler(self, attributes)
