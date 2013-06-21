"""
Helper class for representing a single test.

"""
from quadfloat.attributes import (
    Attributes,
    temporary_attributes,
)
from quadfloat.binary_interchange_format import BinaryInterchangeFormat
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


def identity_conversion(string):
    return string


def binary_conversion(format):
    def convert(hex_character_sequence):
        with temporary_attributes(READ_ATTRIBUTES):
            return format.convert_from_hex_character(hex_character_sequence)
    return convert


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

            lhs, rhs = line.split('->')
            arguments = lhs.split()
            results = rhs.split()
            yield ArithmeticTestCase(
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


# Specific operations.

class TestOperation(object):
    def __init__(self, operation, operand_conversions, result_conversion):
        self.operation = operation
        self.operand_conversions = operand_conversions
        self.result_conversion = result_conversion


def binary_operation_factory(method_name):
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


def convertFromHexCharacter(destination):
    destination_format = binary_format(destination)
    return TestOperation(
        operation=FormatOfOperation(
            destination_format,
            'convert_from_hex_character',
        ),
        operand_conversions=[identity_conversion],
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
            result_conversion=int,
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

    'addition': binary_operation_factory('addition'),
    'subtraction': binary_operation_factory('subtraction'),
    'division': binary_operation_factory('division'),
    'convertFromHexCharacter': convertFromHexCharacter,
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
}
