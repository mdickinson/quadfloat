"""
The :mod:`quadfloat.api` package exports the public classes and functions
for the quadfloat library.

"""
from quadfloat.binary_interchange_format import (
    abs,
    _BinaryFloat,
    BinaryInterchangeFormat,
    class_,
    compare_quiet_equal,
    compare_quiet_greater,
    compare_quiet_greater_equal,
    compare_quiet_greater_unordered,
    compare_quiet_less,
    compare_quiet_less_equal,
    compare_quiet_less_unordered,
    compare_quiet_not_equal,
    compare_quiet_not_greater,
    compare_quiet_not_less,
    compare_quiet_ordered,
    compare_quiet_unordered,
    compare_signaling_equal,
    compare_signaling_greater,
    compare_signaling_greater_equal,
    compare_signaling_less,
    compare_signaling_less_equal,
    compare_signaling_not_equal,
    compare_signaling_not_greater,
    compare_signaling_less_unordered,
    compare_signaling_not_less,
    compare_signaling_greater_unordered,
    convert_to_decimal_character,
    convert_to_hex_character,
    convert_to_integer_exact_ties_to_away,
    convert_to_integer_exact_ties_to_even,
    convert_to_integer_exact_toward_negative,
    convert_to_integer_exact_toward_positive,
    convert_to_integer_exact_toward_zero,
    convert_to_integer_ties_to_away,
    convert_to_integer_ties_to_even,
    convert_to_integer_toward_negative,
    convert_to_integer_toward_positive,
    convert_to_integer_toward_zero,
    copy,
    copy_sign,
    encode,
    is_754_version_1985,
    is_754_version_2008,
    is_canonical,
    is_finite,
    is_infinite,
    is_nan,
    is_normal,
    is_sign_minus,
    is_signaling,
    is_subnormal,
    is_zero,
    log_b,
    max_num,
    max_num_mag,
    min_num,
    min_num_mag,
    negate,
    next_down,
    next_up,
    radix,
    remainder,
    round_to_integral_exact,
    round_to_integral_ties_to_away,
    round_to_integral_ties_to_even,
    round_to_integral_toward_negative,
    round_to_integral_toward_positive,
    round_to_integral_toward_zero,
    scale_b,
    total_order,
    total_order_mag,
)


__all__ = [
    ###########################################################################
    # Binary interchange format
    ###########################################################################
    'BinaryInterchangeFormat',

    # The three basic formats, plus the half-precision format.
    'binary16',
    'binary32',
    'binary64',
    'binary128',

    ###########################################################################
    # The BinaryFloat type
    ###########################################################################
    '_BinaryFloat',

    ###########################################################################
    # Homogeneous operations
    ###########################################################################

    # Note that operations classified as *formatOf* operations are implemented
    # as methods on the BinaryInterchangeFormat objects, so don't need to be
    # imported here.

    # 5.3.1 General operations
    'round_to_integral_ties_to_even',
    'round_to_integral_ties_to_away',
    'round_to_integral_toward_zero',
    'round_to_integral_toward_positive',
    'round_to_integral_toward_negative',
    'round_to_integral_exact',
    'next_up',
    'next_down',
    'remainder',
    'min_num',
    'max_num',
    'min_num_mag',
    'max_num_mag',

    # 5.3.3 logBFormat operations
    'scale_b',
    'log_b',

    # 5.4 formatOf general-computational operations

    # 5.4.1 Arithmetic operations
    'convert_to_integer_ties_to_even',
    'convert_to_integer_toward_zero',
    'convert_to_integer_toward_positive',
    'convert_to_integer_toward_negative',
    'convert_to_integer_ties_to_away',
    'convert_to_integer_exact_ties_to_even',
    'convert_to_integer_exact_toward_zero',
    'convert_to_integer_exact_toward_positive',
    'convert_to_integer_exact_toward_negative',
    'convert_to_integer_exact_ties_to_away',

    # 5.4.2 Conversion operations for floating-point formats and decimal
    # character sequences
    'convert_to_decimal_character',

    # 5.4.3 Conversion operations for binary formats
    'convert_to_hex_character',

    # 5.5 Quiet-computational operations

    # 5.5.1 Sign bit operations
    'copy',
    'negate',
    'abs',
    'copy_sign',

    # 5.6.1 Comparisons
    'compare_quiet_equal',
    'compare_quiet_not_equal',
    'compare_signaling_equal',
    'compare_signaling_greater',
    'compare_signaling_greater_equal',
    'compare_signaling_less',
    'compare_signaling_less_equal',
    'compare_signaling_not_equal',
    'compare_signaling_not_greater',
    'compare_signaling_less_unordered',
    'compare_signaling_not_less',
    'compare_signaling_greater_unordered',
    'compare_quiet_greater',
    'compare_quiet_greater_equal',
    'compare_quiet_less',
    'compare_quiet_less_equal',
    'compare_quiet_unordered',
    'compare_quiet_not_greater',
    'compare_quiet_less_unordered',
    'compare_quiet_not_less',
    'compare_quiet_greater_unordered',
    'compare_quiet_ordered',

    # 5.7 Non-computational operations

    # 5.7.1 Conformance predicates
    'is_754_version_1985',
    'is_754_version_2008',

    # 5.7.2 General operations
    'class_',
    'is_sign_minus',
    'is_normal',
    'is_finite',
    'is_zero',
    'is_subnormal',
    'is_infinite',
    'is_nan',
    'is_signaling',
    'is_canonical',
    'radix',
    'total_order',
    'total_order_mag',

    # 5.7.4 Operations on subsets of flags
    # 'lower_flags',  # Not yet implemented
    # 'raise_flags',  # Not yet implemented
    # 'test_flags',  # Not yet implemented
    # 'test_saved_flags',  # Not yet implemented
    # 'restore_flags',  # Not yet implemented
    # 'save_all_flags',  # Not yet implemented

    # Miscellaneous operations
    'encode',
]

#: Half precision binary floating-point format.
binary16 = BinaryInterchangeFormat(width=16)

#: Single precision binary floating-point format.
binary32 = BinaryInterchangeFormat(width=32)

#: Double precision binary floating-point format.
binary64 = BinaryInterchangeFormat(width=64)

#: Quadruple precision binary floating-point format.
binary128 = BinaryInterchangeFormat(width=128)

#: Testing testing testing
bob = "BOB"
