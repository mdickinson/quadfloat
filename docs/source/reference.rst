Reference
---------

The quadfloat package
+++++++++++++++++++++

.. module:: quadfloat

The :mod:`quadfloat` package is divided into several subpackages.  For the
consumer, the most important of these is the :mod:`quadfloat.api` subpackage.
This contains the public API of the :mod:`quadfloat` package.


The quadfloat.api package
+++++++++++++++++++++++++

.. automodule:: quadfloat.api    

.. autoclass:: quadfloat.api.BinaryInterchangeFormat

.. autoclass:: quadfloat.api._BinaryFloat


The :mod:`quadfloat.api` package exports a few predefined
:class:`BinaryInterchangeFormat` objects, representing the formats described in
section 3.6 of the standard.

.. autodata:: quadfloat.api.binary16

.. autodata:: quadfloat.api.binary32

.. autodata:: quadfloat.api.binary64

.. autodata:: quadfloat.api.binary128
