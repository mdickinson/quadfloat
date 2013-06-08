"""
Tests for attribute mechanism.

"""
from quadfloat.attributes import (
    Attributes,
    get_current_attributes,
    set_current_attributes,
    partial_attributes,
    temporary_attributes,
)
from quadfloat.tests.base_test_case import BaseTestCase


RED, GREEN, BLUE = 'red', 'green', 'blue'
ARTHUR, LANCELOT, MERLIN = 'arthur', 'lancelot', 'merlin'
HOLY_GRAIL = 'to find the holy grail'


class TestAttributes(BaseTestCase):
    def test_creation(self):
        Attributes()

    def test_creation_from_attributes(self):
        attributes = Attributes(
            name=MERLIN,
            favourite_colour=BLUE,
        )
        self.assertEqual(attributes.name, MERLIN)
        self.assertEqual(attributes.favourite_colour, BLUE)
        with self.assertRaises(AttributeError):
            attributes.quest

    def test_push(self):
        first_stack = Attributes(name=MERLIN)
        second_stack = first_stack.push(quest=HOLY_GRAIL)
        third_stack = second_stack.push(name=ARTHUR)

        self.assertEqual(first_stack.name, MERLIN)
        with self.assertRaises(AttributeError):
            first_stack.quest
        self.assertEqual(second_stack.name, MERLIN)
        self.assertEqual(second_stack.quest, HOLY_GRAIL)
        self.assertEqual(third_stack.name, ARTHUR)
        self.assertEqual(third_stack.quest, HOLY_GRAIL)

    def test_temporary_attributes_context_manager(self):
        first_stack = Attributes(name=MERLIN)
        second_stack = Attributes(quest=HOLY_GRAIL)

        with temporary_attributes(first_stack):
            self.assertEqual(get_current_attributes(), first_stack)
            self.assertEqual(get_current_attributes().name, MERLIN)
            with self.assertRaises(AttributeError):
                get_current_attributes().quest

        with temporary_attributes(second_stack):
            self.assertEqual(get_current_attributes(), second_stack)
            self.assertEqual(get_current_attributes().quest, HOLY_GRAIL)
            with self.assertRaises(AttributeError):
                get_current_attributes().name

    def test_temporary_attributes_nested_context_manager(self):
        first_stack = Attributes(name=MERLIN)
        second_stack = Attributes(quest=HOLY_GRAIL)
        with temporary_attributes(first_stack):
            with temporary_attributes(second_stack):
                self.assertEqual(get_current_attributes(), second_stack)
                self.assertEqual(get_current_attributes().quest, HOLY_GRAIL)
                with self.assertRaises(AttributeError):
                    get_current_attributes().name


class TestPartialAttributes(BaseTestCase):
    def setUp(self):
        self.old_attributes = get_current_attributes()
        set_current_attributes(Attributes())

    def tearDown(self):
        set_current_attributes(self.old_attributes)

    def test_partial_attributes_context(self):
        with partial_attributes(favourite_colour=RED):
            self.assertEqual(get_current_attributes().favourite_colour, RED)

        with partial_attributes(favourite_colour=BLUE):
            self.assertEqual(get_current_attributes().favourite_colour, BLUE)

        with partial_attributes(name=ARTHUR):
            self.assertEqual(get_current_attributes().name, ARTHUR)

    def test_nonexistent_attribute(self):
        with self.assertRaises(AttributeError):
            get_current_attributes().favourite_colour

    def test_set_more_than_one_attribute_at_once(self):
        with partial_attributes(name=ARTHUR, favourite_colour=BLUE):
            self.assertEqual(get_current_attributes().name, ARTHUR)
            self.assertEqual(get_current_attributes().favourite_colour, BLUE)

    def test_non_overlapping_nested_contexts(self):
        with partial_attributes(name=ARTHUR):
            with partial_attributes(favourite_colour=BLUE):
                self.assertEqual(get_current_attributes().name, ARTHUR)
                self.assertEqual(
                    get_current_attributes().favourite_colour,
                    BLUE,
                )

    def test_overlapping_nested_contexts(self):
        with partial_attributes(favourite_colour=BLUE):
            self.assertEqual(get_current_attributes().favourite_colour, BLUE)
            with partial_attributes(favourite_colour=RED):
                self.assertEqual(
                    get_current_attributes().favourite_colour,
                    RED,
                )
            # Favourite colour should now be BLUE again.
            self.assertEqual(get_current_attributes().favourite_colour, BLUE)

        # After we leave the with block, favourite_colour is undefined.
        with self.assertRaises(AttributeError):
            get_current_attributes().favourite_colour

    def test_reversion_on_exception(self):
        with self.assertRaises(ZeroDivisionError):
            with partial_attributes(favourite_colour=BLUE):
                self.assertEqual(
                    get_current_attributes().favourite_colour,
                    BLUE,
                )
                1 / 0
        with self.assertRaises(AttributeError):
            get_current_attributes().favourite_colour

        with partial_attributes(favourite_colour=RED):
            with self.assertRaises(ZeroDivisionError):
                with partial_attributes(favourite_colour=BLUE):
                    self.assertEqual(
                        get_current_attributes().favourite_colour,
                        BLUE,
                    )
                    1 / 0
            self.assertEqual(get_current_attributes().favourite_colour, RED)
