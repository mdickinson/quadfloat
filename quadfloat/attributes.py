import contextlib

from quadfloat.exceptions import (
    default_inexact_handler,
    default_invalid_operation_handler,
    default_overflow_handler,
    default_underflow_handler,
)
from quadfloat.rounding_direction import round_ties_to_even

# Attributes.

_attributes = {
    'rounding_direction': round_ties_to_even,
    'inexact_handler': default_inexact_handler,
    'invalid_operation_handler': default_invalid_operation_handler,
    'overflow_handler': default_overflow_handler,
    'underflow_handler': default_underflow_handler,
}


def _current_rounding_direction():
    return _attributes['rounding_direction']


def _current_inexact_handler():
    return _attributes['inexact_handler']


def _current_invalid_operation_handler():
    return _attributes['invalid_operation_handler']


def _current_overflow_handler():
    return _attributes['overflow_handler']


def _current_underflow_handler():
    return _attributes['underflow_handler']


# Context managers to set and restore particular attributes.

@contextlib.contextmanager
def rounding_direction(new_rounding_direction):
    old_rounding_direction = _attributes.get('rounding_direction')
    _attributes['rounding_direction'] = new_rounding_direction
    try:
        yield
    finally:
        _attributes['rounding_direction'] = old_rounding_direction


@contextlib.contextmanager
def inexact_handler(new_handler):
    old_handler = _attributes.get('inexact_handler')
    _attributes['inexact_handler'] = new_handler
    try:
        yield
    finally:
        _attributes['inexact_handler'] = old_handler


@contextlib.contextmanager
def invalid_operation_handler(new_handler):
    old_handler = _attributes.get('invalid_operation_handler')
    _attributes['invalid_operation_handler'] = new_handler
    try:
        yield
    finally:
        _attributes['invalid_operation_handler'] = old_handler


@contextlib.contextmanager
def overflow_handler(new_handler):
    old_handler = _attributes.get('overflow_handler')
    _attributes['overflow_handler'] = new_handler
    try:
        yield
    finally:
        _attributes['overflow_handler'] = old_handler


@contextlib.contextmanager
def underflow_handler(new_handler):
    old_handler = _attributes.get('underflow_handler')
    _attributes['underflow_handler'] = new_handler
    try:
        yield
    finally:
        _attributes['underflow_handler'] = old_handler


# Functions to signal various exceptions.

def _signal_inexact(exception):
    return _current_inexact_handler()(exception)


def _signal_invalid_operation(exception):
    return _current_invalid_operation_handler()(exception)


def _signal_overflow(exception):
    return _current_overflow_handler()(exception)


def _signal_underflow(exception):
    return _current_underflow_handler()(exception)
