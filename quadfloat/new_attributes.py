import contextlib


class _AttributesStack(object):
    def __init__(self, **attrs):
        self._attributes_stack = [
            _PartialAttributes(**attrs)
        ]

    def push(self, attr):
        self._attributes_stack.append(attr)

    def pop(self):
        self._attributes_stack.pop()

    def __getattr__(self, key):
        for partials in reversed(self._attributes_stack):
            try:
                result = getattr(partials, key)
            except AttributeError:
                pass
            else:
                return result
        raise AttributeError(
            "No value is currently set for the attribute {!r}.".format(key))


class _PartialAttributes(object):
    def __new__(cls, **attrs):
        self = object.__new__(cls)
        for attr, value in attrs.items():
            setattr(self, attr, value)
        return self

    @contextlib.contextmanager
    def apply(self, attribute_stack):
        """
        Apply to the given attribute stack.

        """
        attribute_stack.push(self)
        try:
            yield
        finally:
            attribute_stack.pop()


@contextlib.contextmanager
def partial_attributes(**attrs):
    """
    Context manager to temporarily apply a set of attributes to
    the current stack.

    """
    attribute_stack = current_attributes()
    with _PartialAttributes(**attrs).apply(attribute_stack):
        yield


# Private global holding the current attribute stack.
_current_attributes = _AttributesStack()


@contextlib.contextmanager
def attributes(attrs):
    """
    Context manager to temporarily use a different attributes stack.

    """
    global _current_attributes

    stored = _current_attributes
    _current_attributes = attrs
    try:
        yield
    finally:
        _current_attributes = stored

    _current


def current_attributes():
    """
    Return the currently active attribute stack.

    """
    return _current_attributes


# For backwards compatibility.
get_current_attributes = current_attributes
