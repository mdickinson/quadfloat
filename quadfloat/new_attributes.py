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
