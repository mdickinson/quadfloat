import unittest

from quadfloat.rounding_direction import round_ties_to_away, round_ties_to_even
from quadfloat.tininess_detection import BEFORE_ROUNDING, AFTER_ROUNDING


class Attributes(object):
    def __init__(self,
                 rounding_direction,
                 tininess_detection,
                 ):
        if tininess_detection not in (BEFORE_ROUNDING, AFTER_ROUNDING):
            raise ValueError("tininess_detection should be one of {} or {}".format(
                BEFORE_ROUNDING, AFTER_ROUNDING))

        self.rounding_direction = rounding_direction
        self.tininess_detection = tininess_detection


class TestAttributes(unittest.TestCase):
    def test_creation(self):
        attr = Attributes(
            rounding_direction = round_ties_to_even,
            tininess_detection = AFTER_ROUNDING
        )
        self.assertEqual(
            attr.rounding_direction,
            round_ties_to_even,
        )
        self.assertEqual(
            attr.tininess_detection,
            AFTER_ROUNDING,
        )

        attr = Attributes(
            rounding_direction = round_ties_to_away,
            tininess_detection = BEFORE_ROUNDING
        )
        self.assertEqual(
            attr.rounding_direction,
            round_ties_to_away,
        )
        self.assertEqual(
            attr.tininess_detection,
            BEFORE_ROUNDING,
        )

    def test_bad_tininess(self):
        with self.assertRaises(ValueError):
            attr = Attributes(
                rounding_direction = round_ties_to_even,
                tininess_detection = "before"
            )


if __name__ == '__main__':
    unittest.main()
