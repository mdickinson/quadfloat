Introduction
------------

The :mod:`quadfloat` package is intended to provide an implementation for the
binary interchange formats and related binary floating-point operations
described in the IEEE 754 (2008) standard.  The emphasis is on correctness and
ease of verifiability rather than speed; the idea is that this package can
serve as a reference implementation for validation of optimized floating-point
implementations.

The original desire was to provide the quadruple-precision binary128 type
described by the standard.  However, scope creep resulted in a semi-complete
implementation of the IEEE 754 standard.
