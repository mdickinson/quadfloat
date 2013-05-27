"""
Tests for attribute mechanism.

"""
import unittest

from quadfloat.attributes import get_current_attributes, partial_attributes

RED, GREEN, BLUE = 'red', 'green', 'blue'
ARTHUR, LANCELOT = 'arthur', 'lancelot'


class TestPartialAttributes(unittest.TestCase):
    def setUp(self):
        self.attributes = get_current_attributes()

    def test_attributes_context(self):
        with partial_attributes(favourite_colour=RED):
            self.assertEqual(self.attributes.favourite_colour, RED)

        with partial_attributes(favourite_colour=BLUE):
            self.assertEqual(self.attributes.favourite_colour, BLUE)

        with partial_attributes(name=ARTHUR):
            self.assertEqual(self.attributes.name, ARTHUR)

    def test_nonexistent_attribute(self):
        with self.assertRaises(AttributeError):
            self.attributes.favourite_colour

    def test_set_more_than_one_attribute_at_once(self):
        with partial_attributes(name=ARTHUR, favourite_colour=BLUE):
            self.assertEqual(self.attributes.name, ARTHUR)
            self.assertEqual(self.attributes.favourite_colour, BLUE)

    def test_non_overlapping_nested_contexts(self):
        with partial_attributes(name=ARTHUR):
            with partial_attributes(favourite_colour=BLUE):
                self.assertEqual(self.attributes.name, ARTHUR)
                self.assertEqual(self.attributes.favourite_colour, BLUE)

    def test_overlapping_nested_contexts(self):
        with partial_attributes(favourite_colour=BLUE):
            self.assertEqual(self.attributes.favourite_colour, BLUE)
            with partial_attributes(favourite_colour=RED):
                self.assertEqual(self.attributes.favourite_colour, RED)
            # Favourite colour should now be BLUE again.
            self.assertEqual(self.attributes.favourite_colour, BLUE)

        # After we leave the with block, favourite_colour is undefined.
        with self.assertRaises(AttributeError):
            self.attributes.favourite_colour

    def test_reversion_on_exception(self):
        with self.assertRaises(ZeroDivisionError):
            with partial_attributes(favourite_colour=BLUE):
                self.assertEqual(self.attributes.favourite_colour, BLUE)
                1 / 0
        with self.assertRaises(AttributeError):
            self.attributes.favourite_colour

        with partial_attributes(favourite_colour=RED):
            with self.assertRaises(ZeroDivisionError):
                with partial_attributes(favourite_colour=BLUE):
                    self.assertEqual(self.attributes.favourite_colour, BLUE)
                    1 / 0
            self.assertEqual(self.attributes.favourite_colour, RED)


if __name__ == '__main__':
    unittest.main()
