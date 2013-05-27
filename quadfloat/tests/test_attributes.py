"""
Tests for attribute mechanism.

"""
import unittest

from quadfloat.attributes import (
    AttributesStack,
    get_current_attributes,
    set_current_attributes,
    partial_attributes,
    attributes,
)

RED, GREEN, BLUE = 'red', 'green', 'blue'
ARTHUR, LANCELOT, MERLIN = 'arthur', 'lancelot', 'merlin'
HOLY_GRAIL = 'to find the holy grail'


class TestAttributesStack(unittest.TestCase):
    def test_creation(self):
        AttributesStack()

    def test_creation_from_attributes(self):
        attributes = AttributesStack(
            name=MERLIN,
            favourite_colour=BLUE,
        )
        self.assertEqual(attributes.name, MERLIN)
        self.assertEqual(attributes.favourite_colour, BLUE)
        with self.assertRaises(AttributeError):
            attributes.quest

    def test_attributes_context_manager(self):
        first_stack = AttributesStack(name=MERLIN)
        second_stack = AttributesStack(quest=HOLY_GRAIL)

        with attributes(first_stack):
            self.assertEqual(get_current_attributes(), first_stack)
            self.assertEqual(get_current_attributes().name, MERLIN)
            with self.assertRaises(AttributeError):
                get_current_attributes().quest

        with attributes(second_stack):
            self.assertEqual(get_current_attributes(), second_stack)
            self.assertEqual(get_current_attributes().quest, HOLY_GRAIL)
            with self.assertRaises(AttributeError):
                get_current_attributes().name


class TestPartialAttributes(unittest.TestCase):
    def setUp(self):
        self.old_attributes = get_current_attributes()
        self.attributes = AttributesStack()
        set_current_attributes(self.attributes)

    def tearDown(self):
        set_current_attributes(self.old_attributes)

    def test_push_and_pop(self):
        with self.assertRaises(AttributeError):
            self.attributes.favourite_colour

        self.attributes.push(favourite_colour=GREEN)
        self.assertEqual(self.attributes.favourite_colour, GREEN)

        self.attributes.pop()
        with self.assertRaises(AttributeError):
            self.attributes.favourite_colour

        self.attributes.push(favourite_colour=RED)
        self.assertEqual(self.attributes.favourite_colour, RED)

        self.attributes.pop()
        with self.assertRaises(AttributeError):
            self.attributes.favourite_colour

    def test_multiple_push(self):
        self.attributes.push(favourite_colour=GREEN, name=MERLIN)
        self.assertEqual(self.attributes.favourite_colour, GREEN)
        self.assertEqual(self.attributes.name, MERLIN)
        self.attributes.pop()
        with self.assertRaises(AttributeError):
            self.attributes.favourite_colour
        with self.assertRaises(AttributeError):
            self.attributes.name

    def test_nested_push_and_pop(self):
        with self.assertRaises(AttributeError):
            self.attributes.favourite_colour

        self.attributes.push(favourite_colour=GREEN)
        self.assertEqual(self.attributes.favourite_colour, GREEN)

        self.attributes.push(favourite_colour=RED)
        self.assertEqual(self.attributes.favourite_colour, RED)

        self.attributes.pop()
        self.assertEqual(self.attributes.favourite_colour, GREEN)

        self.attributes.pop()
        with self.assertRaises(AttributeError):
            self.attributes.favourite_colour

    def test_partial_attributes_context(self):
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
