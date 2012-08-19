"""
Python 2 / 3 compatibility code.

"""
import sys

if sys.version_info.major == 2:
    from future_builtins import map as _map
    from future_builtins import zip as _zip

else:
    _map = map
    _zip = zip
