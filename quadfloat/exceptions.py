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
    def __init__(self):
        pass

    def default_handler(self):
        raise ValueError("Invalid operation returning an integer.")


class InvalidBooleanOperationException(object):
    def __init__(self):
        pass

    def default_handler(self):
        raise ValueError("Invalid operation returning a boolean.")


class SignalingNaNException(object):
    """
    InvalidOperation exception thrown as a result of an arithmetic
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
    Exception thrown when a result is inexact.

    """
    def __init__(self, rounded):
        self.rounded = rounded

    def default_handler(self):
        return self.rounded


def default_invalid_operation_handler(exception):
    return exception.default_handler()


def default_inexact_handler(exception):
    return exception.default_handler()
