import re
import unittest


_MAX_LENGTH = 80


def safe_repr(obj, short=False):
    try:
        result = repr(obj)
    except Exception:
        result = object.__repr__(obj)
    if not short or len(result) < _MAX_LENGTH:
        return result
    return result[:_MAX_LENGTH] + ' [truncated]...'


class _AssertRaisesBaseContext(object):

    def __init__(self, expected, test_case, callable_obj=None,
                 expected_regex=None):
        self.expected = expected
        self.test_case = test_case
        if callable_obj is not None:
            try:
                self.obj_name = callable_obj.__name__
            except AttributeError:
                self.obj_name = str(callable_obj)
        else:
            self.obj_name = None
        if isinstance(expected_regex, (bytes, str)):
            expected_regex = re.compile(expected_regex)
        self.expected_regex = expected_regex
        self.msg = None

    def _raiseFailure(self, standardMsg):
        msg = self.test_case._formatMessage(self.msg, standardMsg)
        raise self.test_case.failureException(msg)

    def handle(self, name, callable_obj, args, kwargs):
        """
        If callable_obj is None, assertRaises/Warns is being used as a
        context manager, so check for a 'msg' kwarg and return self.
        If callable_obj is not None, call it passing args and kwargs.
        """
        if callable_obj is None:
            self.msg = kwargs.pop('msg', None)
            return self
        with self:
            callable_obj(*args, **kwargs)


class _AssertRaisesContext(_AssertRaisesBaseContext):
    """A context manager used to implement TestCase.assertRaises* methods."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            try:
                exc_name = self.expected.__name__
            except AttributeError:
                exc_name = str(self.expected)
            if self.obj_name:
                self._raiseFailure(
                    "{0} not raised by {1}".format(exc_name,
                                                   self.obj_name))
            else:
                self._raiseFailure("{0} not raised".format(exc_name))
        if not issubclass(exc_type, self.expected):
            # let unexpected exceptions pass through
            return False

        self.exception = exc_value
        if self.expected_regex is None:
            return True

        expected_regex = self.expected_regex
        if not expected_regex.search(str(exc_value)):
            self._raiseFailure(
                '"{0}" does not match "{1}"'.format(
                    expected_regex.pattern, str(exc_value)))
        return True


def _identifying_string(binary_float):
    fmt = binary_float.format
    return "{0} (format {1})".format(
        fmt.convert_to_hex_character_simple(binary_float),
        binary_float.format,
    )


class BaseTestCase(unittest.TestCase):
    # For Python versions <= 2.6.
    def assertRaises(self, excClass, callableObj=None, *args, **kwargs):
        """Fail unless an exception of class excClass is raised
           by callableObj when invoked with arguments args and keyword
           arguments kwargs. If a different type of exception is
           raised, it will not be caught, and the test case will be
           deemed to have suffered an error, exactly as for an
           unexpected exception.

           If called with callableObj omitted or None, will return a
           context object used like this::

                with self.assertRaises(SomeException):
                    do_something()

           An optional keyword argument 'msg' can be provided when assertRaises
           is used as a context object.

           The context manager keeps a reference to the exception as
           the 'exception' attribute. This allows you to inspect the
           exception after the assertion::

               with self.assertRaises(SomeException) as cm:
                   do_something()
               the_exception = cm.exception
               self.assertEqual(the_exception.error_code, 3)
        """
        context = _AssertRaisesContext(excClass, self, callableObj)
        return context.handle('assertRaises', callableObj, args, kwargs)

    def assertIsInstance(self, obj, cls, msg=None):
        """Same as self.assertTrue(isinstance(obj, cls)), with a nicer
        default message."""
        if not isinstance(obj, cls):
            standardMsg = '%s is not an instance of %r' % (safe_repr(obj), cls)
            self.fail(self._formatMessage(msg, standardMsg))

    def assertLess(self, a, b, msg=None):
        """Like self.assertTrue(a < b), but with a nicer default message."""
        if not a < b:
            standardMsg = '%s not less than %s' % (safe_repr(a), safe_repr(b))
            self.fail(self._formatMessage(msg, standardMsg))

    def assertInterchangeable(self, quad1, quad2, msg=None):
        """
        Assert that two _BinaryFloat instances are interchangeable.

        This means more than just being numerically equal:  for example, -0.0
        and 0.0 are equal, but not interchangeable.

        """
        self.assertEqual(
            _identifying_string(quad1),
            _identifying_string(quad2),
            msg)

    def assertNotInterchangeable(self, quad1, quad2, msg=None):
        """
        Assert that two _BinaryFloat instances are not interchangeable.

        See also: assertInterchangeable

        """
        self.assertNotEqual(
            _identifying_string(quad1),
            _identifying_string(quad2),
            msg)
