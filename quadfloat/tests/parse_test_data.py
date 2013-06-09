"""
Helper class for representing a single test.

"""
from quadfloat import binary16, binary32, binary64, binary128
from quadfloat.attributes import (
    Attributes,
    temporary_attributes,
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

from quadfloat.tests.arithmetic_test_case import ArithmeticTestCase

formats = {
    'binary16': binary16,
    'binary32': binary32,
    'binary64': binary64,
    'binary128': binary128,
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


# Attributes used when reading a RHS.


def raising_inexact_handler(exc):
    raise ValueError(
        "Got inexact value; rounded value is {0}".format(exc.rounded)
    )


READ_ATTRIBUTES = Attributes(
    rounding_direction=round_ties_to_even,
    tininess_detection=AFTER_ROUNDING,
    # Handlers for inexact, underflow.  We shouldn't encounter overflow.
    inexact_handler=raising_inexact_handler,
    underflow_handler=UnderflowException.default_handler,
)


def parse_test_data(test_content):
    """Given a string containing test data, generate ArithmeticTestCase objects
    representing the individual tests.

    """
    lines = test_content.splitlines()

    current_operation = None
    current_operation_attributes = {}

    # Attributes to apply to tests.
    attributes = Attributes()

    for line in lines:
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
            operation_factory = operations[current_operation]
            operation = operation_factory(**current_operation_attributes)

            args, results = line.split('->')

            arguments = args.split()
            results = results.split()

            #with temporary_attributes(READ_ATTRIBUTES):
            #    result = result_format.convert_from_hex_character(results[0])
            yield ArithmeticTestCase(
                args=[
                    argument_conversion(argument)
                    for argument_conversion, argument in zip(
                        operation.argument_conversions,
                        arguments,
                    )
                ],
                result=operation.result_conversion(results[0]),
                flags=set(results[1:]),
                operation=operation,
                attributes=attributes,
            )


# Specific operations.

class addition(object):
    def __init__(self, destination, source1, source2):
        self.destination = destination
        self.source1 = source1
        self.source2 = source2
        self._destination_format = formats[self.destination]
        self.argument_conversions = [
            binary_conversion(formats[self.source1]),
            binary_conversion(formats[self.source2]),
        ]
        self.result_conversion = binary_conversion(formats[self.destination])
        self.__name__ = "addition"

    def __call__(self, *args):
        return self._destination_format.addition(*args)


class subtraction(object):
    def __init__(self, destination, source1, source2):
        self.destination = destination
        self.source1 = source1
        self.source2 = source2
        self._destination_format = formats[self.destination]
        self.argument_conversions = [
            binary_conversion(formats[self.source1]),
            binary_conversion(formats[self.source2]),
        ]
        self.result_conversion = binary_conversion(formats[self.destination])
        self.__name__ = "subtraction"

    def __call__(self, *args):
        return self._destination_format.subtraction(*args)


class roundToIntegralTiesToAway(object):
    def __init__(self, source):
        self.source = source
        self.argument_conversions = [binary_conversion(formats[self.source])]
        self.result_conversion = binary_conversion(formats[self.source])
        self.__name__ = "roundToIntegralTiesToAway"

    def __call__(self, *args):
        arg, = args
        return arg.round_to_integral_ties_to_away()


class convertFromHexCharacter(object):
    def __init__(self, destination):
        self.destination = destination
        self._destination_format = formats[self.destination]
        # Strings do not need to be converted.
        self.argument_conversions = [lambda x: x]
        # XXX Use .name, not .__name__
        self.__name__ = '{}-convertFromHexCharacter'.format(destination)
        self.result_conversion = binary_conversion(formats[self.destination])

    def __call__(self, *args):
        return self._destination_format.convert_from_hex_character(*args)


operations = {
    'addition': addition,
    'subtraction': subtraction,
    'convertFromHexCharacter': convertFromHexCharacter,
    'roundToIntegralTiesToAway': roundToIntegralTiesToAway
}
