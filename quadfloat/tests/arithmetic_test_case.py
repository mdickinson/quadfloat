"""
Helper class for representing a single test.

"""
from quadfloat import binary16, binary32, binary64, binary128
from quadfloat.attributes import Attributes, temporary_attributes
from quadfloat.exceptions import UnderflowException
from quadfloat.rounding_direction import round_ties_to_away, round_ties_to_even
from quadfloat.tininess_detection import BEFORE_ROUNDING, AFTER_ROUNDING


class ArithmeticTestCase(object):
    def __init__(self, args, result, flags, callable, attributes):
        self.args = args
        self.result = result
        self.flags = flags
        self.callable = callable
        self.attributes = attributes

    def __repr__(self):
        return "{} {} -> {} {} {}".format(
            self.callable.__name__,
            self.args,
            self.result,
            self.flags,
            self.attributes,
        )


# Mapping from operations to callables.

def convertFromHexCharacter(destination):
    destination_format = formats[destination]
    return destination_format.convert_from_hex_character


formats = {
    'binary16': binary16,
    'binary32': binary32,
    'binary64': binary64,
    'binary128': binary128,
}


operations = {
    'convertFromHexCharacter': convertFromHexCharacter,
}


tininess_detection_modes = {
    'after-rounding': AFTER_ROUNDING,
    'before-rounding': BEFORE_ROUNDING,
}


rounding_directions = {
    'round-ties-to-away': round_ties_to_away,
    'round-ties-to-even': round_ties_to_even,
}


# Attributes used when reading a RHS.
READ_ATTRIBUTES = Attributes(
    rounding_direction=round_ties_to_even,
    tininess_detection=AFTER_ROUNDING,
    underflow_handler=UnderflowException.default_handler,
)


def parse_test_data(test_content):
    """Given a string containing test data, generate ArithmeticTestCase objects
    representing the individual tests.

    """
    lines = test_content.splitlines()

    current_operation = None
    current_operation_attributes = {}

    attributes = Attributes(
        rounding_direction=round_ties_to_even,
        tininess_detection=AFTER_ROUNDING,
    )

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
                    raise ValueError("Unrecognized attribute: {}".format(lhs))
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
                raise ValueError("Unsupported directive: {}".format(line))
        else:
            args, results = line.split('->')
            args = args.split()
            results = results.split()
            # XXX Check that conversion is exact.
            result_format = formats[
                current_operation_attributes['destination']]
            with temporary_attributes(READ_ATTRIBUTES):
                result = result_format.convert_from_hex_character(results[0])
            flags = set(results[1:])
            operation_factory = operations[current_operation]
            operation = operation_factory(**current_operation_attributes)
            yield ArithmeticTestCase(
                args=args,
                result=result,
                flags=flags,
                # Function to call should be encoded in test file.
                callable=operation,
                attributes=attributes,
            )
