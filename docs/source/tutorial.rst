Tutorial
--------

Start by importing everything from :mod:`quadfloat.api`::

    from quadfloat.api import *

This brings in constants :data:`~quadfloat.api.binary16`,
:data:`~quadfloat.api.binary32`, :data:`~quadfloat.api.binary64` and
:data:`~quadfloat.api.binary128` representing the corresponding binary
interchange formats.  These formats are callable, and may be used to generate
floating-point numbers.

    >>> from quadfloat.api import *
    >>> x = binary128('1.1')  # Create from a decimal string.
    >>> print x
    1.1
    >>> y = binary128(13.5)  # Create from a Python float.
    >>> print x + y
    14.6

The IEEE 754 computational operations are divided into two main classes:
"formatOf" operations and "homogeneous" operations.  The "formatOf" operations
are represented as methods on format objects.  The :func:`square_root`
operation is one example.

    >>> print binary32.square_root(x)
    1.0488088
    >>> print binary128.square_root(x)
    1.0488088481701515469914535136799376

The homogenous operations are represented by module-level functions.  For example:

    >>> print next_up(x)
    1.1000000000000000000000000000000003

Note that there's a common type :class:`~quadfloat.api._BinaryFloat` for all binary
floating-point numbers, regardless of their format.
