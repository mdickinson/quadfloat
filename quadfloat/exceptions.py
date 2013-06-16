"""
Standard exceptions.

"""


class InvalidOperationException(object):
    """
    Class representing an InvalidOperation exception.

    """
    def __init__(self, format):
        self.format = format

    def default_handler(self):
        return self.format._nan(
            sign=False,
            signaling=False,
            payload=0,
        )


class InvalidIntegerOperationException(object):
    """
    Class representing an invalid operation that returns an integer.

    """
    def __init__(self, payload):
        # 'payload' is the value to return in non-stop mode.
        self.payload = payload

    def default_handler(self):
        return self.payload


class InvalidBooleanOperationException(object):
    def __init__(self):
        pass

    def default_handler(self):
        raise ValueError("Invalid operation returning a boolean.")


class SignalingNaNException(object):
    """
    InvalidOperation exception signaled as a result of an arithmetic
    operation encountering a signaling NaN.

    """
    def __init__(self, format, snan):
        self.format = format
        # The signaling NaN that caused this exception.
        self.snan = snan

    def default_handler(self):
        return self.format(self.snan._quieten_nan())


class InexactException(object):
    """
    Exception signaled when a result is inexact.

    """
    def __init__(self, rounded):
        self.rounded = rounded

    def default_handler(self):
        return self.rounded


class UnderflowException(object):
    """
    Exception signaled when a result is non-zero and tiny.

    """
    def __init__(self, rounded, inexact):
        self.rounded = rounded
        self.inexact = inexact

    def default_handler(self):
        # Local import to avoid circular imports.
        from quadfloat.attributes import _signal_inexact
        if self.inexact:
            return _signal_inexact(InexactException(self.rounded))
        else:
            return self.rounded


class OverflowException(object):
    """
    Exception signaled when a result overflows.

    """
    def __init__(self, rounded):
        self.rounded = rounded

    def default_handler(self):
        # Local import to avoid circular imports.
        from quadfloat.attributes import _signal_inexact
        return _signal_inexact(InexactException(self.rounded))


class DivideByZeroException(object):
    """
    Exception signaled when an operation with finite operands
    results in an exact infinity.

    """
    def __init__(self, sign, format):
        self.sign = sign
        self.format = format

    def default_handler(self):
        return self.format._infinite(self.sign)
