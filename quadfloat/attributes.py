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

from quadfloat.new_attributes import (
    get_current_attributes,
    partial_attributes,
    _AttributesStack,
    _PartialAttributes,
)


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
