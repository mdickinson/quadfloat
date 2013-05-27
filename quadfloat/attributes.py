"""
Support for IEEE 754 attributes.  These are settings that may affect the output
or operation of various functions.

IEEE 754 requires the following attributes:

 - rounding-direction attributes

and recommends:

 - alternate exception handling attributes

There may also be:

 - preferred width attributes
 - value-changing optimization attributes
 - reproducibility attributes

We add one more attribute not included in the standard:

 - tininess detection

"""
import contextlib


class _AttributesStack(object):
    def __init__(self, **attrs):
        self._attributes_stack = [attrs]

    def push(self, **attrs):
        self._attributes_stack.append(attrs)

    def pop(self):
        self._attributes_stack.pop()

    def __getattr__(self, key):
        for partials in reversed(self._attributes_stack):
            try:
                result = partials[key]
            except KeyError:
                pass
            else:
                return result
        raise AttributeError(
            "No value is currently set for the attribute {!r}.".format(key))


# Private global holding the current attribute stack.
_current_attributes = _AttributesStack()


def get_current_attributes():
    """
    Return the currently active attribute stack.

    """
    return _current_attributes


def set_current_attributes(attrs):
    """
    Set the currently active attributes.

    """
    global _current_attributes
    _current_attributes = attrs


@contextlib.contextmanager
def attributes(attrs):
    """
    Context manager to temporarily use a different attributes stack.

    """
    stored = get_current_attributes()
    set_current_attributes(attrs)
    try:
        yield
    finally:
        set_current_attributes(stored)


@contextlib.contextmanager
def partial_attributes(**attrs):
    """
    Context manager to temporarily apply given attributes to the current stack.

    """
    attribute_stack = get_current_attributes()
    attribute_stack.push(**attrs)
    try:
        yield
    finally:
        attribute_stack.pop()


# Helper functions.

def _current_inexact_handler():
    return get_current_attributes().inexact_handler


def _current_invalid_operation_handler():
    return get_current_attributes().invalid_operation_handler


def _current_overflow_handler():
    return get_current_attributes().overflow_handler


def _current_underflow_handler():
    return get_current_attributes().underflow_handler


# Context managers to set and restore particular attributes.

@contextlib.contextmanager
def rounding_direction(new_rounding_direction):
    with partial_attributes(rounding_direction=new_rounding_direction):
        yield


@contextlib.contextmanager
def inexact_handler(new_handler):
    with partial_attributes(inexact_handler=new_handler):
        yield


@contextlib.contextmanager
def invalid_operation_handler(new_handler):
    with partial_attributes(invalid_operation_handler=new_handler):
        yield


@contextlib.contextmanager
def overflow_handler(new_handler):
    with partial_attributes(overflow_handler=new_handler):
        yield


@contextlib.contextmanager
def underflow_handler(new_handler):
    with partial_attributes(underflow_handler=new_handler):
        yield


# Functions to signal various exceptions.

def _signal_inexact(exception):
    return _current_inexact_handler()(exception)


def _signal_invalid_operation(exception):
    return _current_invalid_operation_handler()(exception)


def _signal_overflow(exception):
    return _current_overflow_handler()(exception)


def _signal_underflow(exception):
    return _current_underflow_handler()(exception)
