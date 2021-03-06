"""
Helper class for representing a single test.

"""
import json

from quadfloat.api import BinaryInterchangeFormat
from quadfloat.attributes import (
    Attributes,
    temporary_attributes,
)
from quadfloat.bit_string import (
    BitString,
)
from quadfloat.exceptions import (
    UnderflowException,
)
from quadfloat.rounding_direction import (
    round_ties_to_away,
    round_ties_to_even,
    round_toward_positive,
    round_toward_negative,
    round_toward_zero,
)
from quadfloat.tininess_detection import BEFORE_ROUNDING, AFTER_ROUNDING

from quadfloat.tests.arithmetic_test_case import (
    ArithmeticTestCase,
    ArithmeticTestResult,
    FormatOfOperation,
    HomogeneousOperation,
)


def binary_format(key):
    if not key.startswith('binary'):
        raise ValueError("Invalid format string: {}".format(key))
    width = int(key[len('binary'):])
    return BinaryInterchangeFormat(width=width)


formats = {
    'binary16': BinaryInterchangeFormat(width=16),
    'binary32': BinaryInterchangeFormat(width=32),
    'binary64': BinaryInterchangeFormat(width=64),
    'binary128': BinaryInterchangeFormat(width=128),
    'binary1024': BinaryInterchangeFormat(width=1024),
}


tininess_detection_modes = {
    'afterRounding': AFTER_ROUNDING,
    'beforeRounding': BEFORE_ROUNDING,
}


rounding_directions = {
    'roundTiesToAway': round_ties_to_away,
    'roundTiesToEven': round_ties_to_even,
    'roundTowardPositive': round_toward_positive,
    'roundTowardNegative': round_toward_negative,
    'roundTowardZero': round_toward_zero,
}


def binary_conversion(format):
    def convert(hex_character_sequence):
        with temporary_attributes(READ_ATTRIBUTES):
            return format.convert_from_hex_character(hex_character_sequence)
    return convert


def bool_conversion(string):
    string = string.lower()
    if string == 'true':
        return True
    elif string == 'false':
        return False
    else:
        raise ValueError("Can't interpret {!r} as a boolean.".format(string))


def json_str(string):
    """
    Convert a Python string representing a JSON quoted string into
    the underlying string object.

    """
    return json.loads(string)


def bool_or_value_error(string):
    try:
        return bool_conversion(string)
    except ValueError:
        pass

    if string == "ValueError":
        return string
    else:
        raise ValueError("Can't interpret {!r}".format(string))


def int_or_value_error(string):
    try:
        return int(string)
    except ValueError:
        pass

    if string == "ValueError":
        return string
    else:
        raise ValueError("Can't interpret {!r}.".format(string))


# Attributes used when reading a RHS.

def raising_inexact_handler(exception, attributes):
    raise ValueError(
        "Got inexact value; rounded value is {0}".format(exception.rounded)
    )


READ_ATTRIBUTES = Attributes(
    rounding_direction=round_ties_to_even,
    tininess_detection=AFTER_ROUNDING,
    # Handlers for inexact, underflow.  We shouldn't encounter overflow.
    inexact_handler=raising_inexact_handler,
    underflow_handler=UnderflowException.default_handler,
    flag_set=set(),
)


def parse_test_data(test_content, source_file):
    """Given a string containing test data, generate ArithmeticTestCase objects
    representing the individual tests.

    """
    lines = test_content.splitlines()

    current_operation = None
    current_operation_attributes = {}

    # Attributes to apply to tests.
    attributes = Attributes()

    for line_number, line in enumerate(lines, start=1):
        # Strip comments; skip blank lines.
        if '#' in line:
            line = line[:line.index('#')]
        line = line.strip()
        if not line:
            continue

        # Directives.
        if ':' in line:

            # Deal with attributes.
            if line.startswith('attribute'):
                line = line[len('attribute'):]
                lhs, rhs = [piece.strip() for piece in line.split(':')]

                if lhs == 'rounding-direction':
                    attributes = attributes.push(
                        rounding_direction=rounding_directions[rhs]
                    )
                elif lhs == 'tininess-detection':
                    attributes = attributes.push(
                        tininess_detection=tininess_detection_modes[rhs]
                    )
                else:
                    raise ValueError("Unrecognized attribute: {0}".format(lhs))
            elif line.startswith('operation'):
                line = line[len('operation'):]
                lhs, rhs = [piece.strip() for piece in line.split(':')]
                if not lhs:
                    current_operation = rhs
                    current_operation_attributes = {}
                elif current_operation is None:
                    raise ValueError("Can't specify attribute before "
                                     "operation")
                else:
                    current_operation_attributes[lhs] = rhs
            else:
                raise ValueError("Unsupported directive: {0}".format(line))
        else:
            operation_factory = operation_factories[current_operation]
            operation = operation_factory(**current_operation_attributes)

            try:
                lhs, rhs = line.split('->')
                arguments = lhs.split()
                results = rhs.split()
                test_case = ArithmeticTestCase(
                    operands=[
                        operand_conversion(argument)
                        for operand_conversion, argument in zip(
                            operation.operand_conversions,
                            arguments,
                        )
                    ],
                    expected_result=ArithmeticTestResult(
                        result=operation.result_conversion(results[0]),
                        flags=set(results[1:]),
                    ),
                    operation=operation.operation,
                    attributes=attributes,
                    line_number=line_number,
                    source_file=source_file,
                )
            except ValueError as e:
                raise ValueError(
                    "Error generating testcase from "
                    "file {}, line {}: {}".format(source_file, line_number, e))
            yield test_case


# Specific operations.

class TestOperation(object):
    def __init__(self, operation, operand_conversions, result_conversion):
        self.operation = operation
        self.operand_conversions = operand_conversions
        self.result_conversion = result_conversion


def unary_formatof_operation(method_name):
    """
    Factory for unary formatOf operations.

    """
    def operation_factory(destination, source):
        """
        Factory for unary formatOf operations.

        """
        destination_format = binary_format(destination)
        source_format = binary_format(source)
        return TestOperation(
            operation=FormatOfOperation(destination_format, method_name),
            operand_conversions=[binary_conversion(source_format)],
            result_conversion=binary_conversion(destination_format),
        )
    return operation_factory


def binary_formatof_operation(method_name):
    """
    Factory for binary formatOf operations.

    """
    def operation_factory(destination, source1, source2):
        """
        Factory for binary formatOf operations.

        """
        destination_format = binary_format(destination)
        source1_format = binary_format(source1)
        source2_format = binary_format(source2)
        return TestOperation(
            operation=FormatOfOperation(destination_format, method_name),
            operand_conversions=[
                binary_conversion(source1_format),
                binary_conversion(source2_format),
            ],
            result_conversion=binary_conversion(destination_format),
        )
    return operation_factory


def ternary_formatof_operation(method_name):
    """
    Factory for ternary formatOf operations.

    """
    def operation_factory(destination, source1, source2, source3):
        """
        Factory for ternary formatOf operations.

        """
        destination_format = binary_format(destination)
        source1_format = binary_format(source1)
        source2_format = binary_format(source2)
        source3_format = binary_format(source3)
        return TestOperation(
            operation=FormatOfOperation(destination_format, method_name),
            operand_conversions=[
                binary_conversion(source1_format),
                binary_conversion(source2_format),
                binary_conversion(source3_format),
            ],
            result_conversion=binary_conversion(destination_format),
        )
    return operation_factory


def comparison(method_name):
    def comparison_factory(source1, source2):
        source1_format = binary_format(source1)
        source2_format = binary_format(source2)
        return TestOperation(
            operation=HomogeneousOperation(method_name),
            operand_conversions=[
                binary_conversion(source1_format),
                binary_conversion(source2_format),
            ],
            result_conversion=bool_or_value_error,
        )
    return comparison_factory


def unary_source_operation(method_name):
    def unary_operation_factory(source):
        source_format = binary_format(source)
        return TestOperation(
            operation=HomogeneousOperation(method_name),
            operand_conversions=[binary_conversion(source_format)],
            result_conversion=binary_conversion(source_format),
        )
    return unary_operation_factory


def binary_source_operation(method_name):
    def binary_operation_factory(source):
        source_format = binary_format(source)
        return TestOperation(
            operation=HomogeneousOperation(method_name),
            operand_conversions=[binary_conversion(source_format)] * 2,
            result_conversion=binary_conversion(source_format),
        )
    return binary_operation_factory


def convert_from_hex_character(destination):
    destination_format = binary_format(destination)
    return TestOperation(
        operation=FormatOfOperation(
            destination_format,
            'convert_from_hex_character',
        ),
        operand_conversions=[json_str],
        result_conversion=binary_conversion(destination_format),
    )


def convert_from_decimal_character(destination):
    destination_format = binary_format(destination)
    return TestOperation(
        operation=FormatOfOperation(
            destination_format,
            'convert_from_decimal_character',
        ),
        operand_conversions=[json_str],
        result_conversion=binary_conversion(destination_format),
    )


def scaleB(source):
    source_format = binary_format(source)
    return TestOperation(
        operation=HomogeneousOperation('scale_b'),
        operand_conversions=[binary_conversion(source_format), int],
        result_conversion=binary_conversion(source_format),
    )


def logB(source):
    source_format = binary_format(source)
    return TestOperation(
        operation=HomogeneousOperation('log_b'),
        operand_conversions=[binary_conversion(source_format)],
        result_conversion=int,
    )


def convert_from_int(destination):
    destination_format = binary_format(destination)
    return TestOperation(
        operation=FormatOfOperation(destination_format, 'convert_from_int'),
        operand_conversions=[int],
        result_conversion=binary_conversion(destination_format),
    )


def convert_to_integer(name):
    def integer_operation_factory(source):
        source_format = binary_format(source)
        return TestOperation(
            operation=HomogeneousOperation(name),
            operand_conversions=[binary_conversion(source_format)],
            result_conversion=int_or_value_error,
        )
    return integer_operation_factory


def convert_format(source, destination):
    destination_format = binary_format(destination)
    source_format = binary_format(source)
    return TestOperation(
        operation=FormatOfOperation(destination_format, 'convert_format'),
        operand_conversions=[binary_conversion(source_format)],
        result_conversion=binary_conversion(destination_format),
    )


def convert_to_hex_character(source):
    source_format = binary_format(source)
    return TestOperation(
        operation=HomogeneousOperation('convert_to_hex_character'),
        operand_conversions=[binary_conversion(source_format), json_str],
        result_conversion=json_str,
    )


def convert_to_decimal_character(source):
    source_format = binary_format(source)
    return TestOperation(
        operation=HomogeneousOperation('convert_to_decimal_character'),
        operand_conversions=[binary_conversion(source_format), json_str],
        result_conversion=json_str,
    )


def total_order(source):
    source_format = binary_format(source)
    return TestOperation(
        operation=HomogeneousOperation('total_order'),
        operand_conversions=[binary_conversion(source_format)] * 2,
        result_conversion=bool_conversion,
    )


def total_order_mag(source):
    source_format = binary_format(source)
    return TestOperation(
        operation=HomogeneousOperation('total_order_mag'),
        operand_conversions=[binary_conversion(source_format)] * 2,
        result_conversion=bool_conversion,
    )


def class_(source):
    source_format = binary_format(source)
    return TestOperation(
        operation=HomogeneousOperation('class_'),
        operand_conversions=[binary_conversion(source_format)],
        result_conversion=str,
    )


def unary_predicate(method_name):
    def unary_predicate_factory(source):
        source_format = binary_format(source)
        return TestOperation(
            operation=HomogeneousOperation(method_name),
            operand_conversions=[binary_conversion(source_format)],
            result_conversion=bool_conversion,
        )
    return unary_predicate_factory


def nullary_predicate(method_name):
    def nullary_predicate_factory():
        return TestOperation(
            operation=HomogeneousOperation(method_name),
            operand_conversions=[],
            result_conversion=bool_conversion,
        )
    return nullary_predicate_factory


def radix(source):
    source_format = binary_format(source)
    return TestOperation(
        operation=HomogeneousOperation('radix'),
        operand_conversions=[binary_conversion(source_format)],
        result_conversion=int,
    )


def decode(destination):
    destination_format = binary_format(destination)
    return TestOperation(
        operation=FormatOfOperation(
            destination_format,
            'decode',
        ),
        operand_conversions=[BitString],
        result_conversion=binary_conversion(destination_format),
    )


def encode(source):
    source_format = binary_format(source)
    return TestOperation(
        operation=HomogeneousOperation('encode'),
        operand_conversions=[binary_conversion(source_format)],
        result_conversion=BitString,
    )


_uso = unary_source_operation
operation_factories = {
    # 5.3.1 General operations
    'roundToIntegralTiesToEven': _uso('round_to_integral_ties_to_even'),
    'roundToIntegralTiesToAway': _uso('round_to_integral_ties_to_away'),
    'roundToIntegralTowardZero': _uso('round_to_integral_toward_zero'),
    'roundToIntegralTowardPositive': _uso('round_to_integral_toward_positive'),
    'roundToIntegralTowardNegative': _uso('round_to_integral_toward_negative'),
    'roundToIntegralExact': _uso('round_to_integral_exact'),
    'nextUp': unary_source_operation('next_up'),
    'nextDown': unary_source_operation('next_down'),
    'remainder': binary_source_operation('remainder'),
    'minNum': binary_source_operation('min_num'),
    'maxNum': binary_source_operation('max_num'),
    'minNumMag': binary_source_operation('min_num_mag'),
    'maxNumMag': binary_source_operation('max_num_mag'),

    'addition': binary_formatof_operation('addition'),
    'subtraction': binary_formatof_operation('subtraction'),
    'multiplication': binary_formatof_operation('multiplication'),
    'division': binary_formatof_operation('division'),
    'squareRoot': unary_formatof_operation('square_root'),
    'fusedMultiplyAdd': ternary_formatof_operation('fused_multiply_add'),
    'convertFromHexCharacter': convert_from_hex_character,
    'convertToHexCharacter': convert_to_hex_character,
    'convertFromDecimalCharacter': convert_from_decimal_character,
    'convertToDecimalCharacter': convert_to_decimal_character,
    'scaleB': scaleB,
    'logB': logB,

    'convertToIntegerTiesToEven': convert_to_integer(
        'convert_to_integer_ties_to_even'),
    'convertToIntegerTiesToAway': convert_to_integer(
        'convert_to_integer_ties_to_away'),
    'convertToIntegerTowardZero': convert_to_integer(
        'convert_to_integer_toward_zero'),
    'convertToIntegerTowardPositive': convert_to_integer(
        'convert_to_integer_toward_positive'),
    'convertToIntegerTowardNegative': convert_to_integer(
        'convert_to_integer_toward_negative'),
    'convertToIntegerExactTiesToEven': convert_to_integer(
        'convert_to_integer_exact_ties_to_even'),
    'convertToIntegerExactTiesToAway': convert_to_integer(
        'convert_to_integer_exact_ties_to_away'),
    'convertToIntegerExactTowardZero': convert_to_integer(
        'convert_to_integer_exact_toward_zero'),
    'convertToIntegerExactTowardPositive': convert_to_integer(
        'convert_to_integer_exact_toward_positive'),
    'convertToIntegerExactTowardNegative': convert_to_integer(
        'convert_to_integer_exact_toward_negative'),

    'convertFromInt': convert_from_int,
    'convertFormat': convert_format,
    'totalOrder': total_order,
    'totalOrderMag': total_order_mag,
    'class': class_,

    'isCanonical': unary_predicate('is_canonical'),
    'isFinite': unary_predicate('is_finite'),
    'isInfinite': unary_predicate('is_infinite'),
    'isNaN': unary_predicate('is_nan'),
    'isNormal': unary_predicate('is_normal'),
    'isSignMinus': unary_predicate('is_sign_minus'),
    'isSignaling': unary_predicate('is_signaling'),
    'isSubnormal': unary_predicate('is_subnormal'),
    'isZero': unary_predicate('is_zero'),
    'radix': radix,

    'abs': unary_source_operation('abs'),
    'copy': unary_source_operation('copy'),
    'negate': unary_source_operation('negate'),
    'copySign': binary_source_operation('copy_sign'),

    'compareQuietEqual': comparison('compare_quiet_equal'),
    'compareQuietLess': comparison('compare_quiet_less'),
    'compareQuietLessEqual': comparison('compare_quiet_less_equal'),

    'is754version1985': nullary_predicate('is_754_version_1985'),
    'is754version2008': nullary_predicate('is_754_version_2008'),

    'decode': decode,
    'encode': encode,
}
