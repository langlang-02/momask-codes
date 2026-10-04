import unittest
from g1_filter_flat_ground import classify


class DescriptionFilterTest(unittest.TestCase):
    def test_external_support(self):
        for text in ['get down from 50 cm box', 'jump backwards off a 50 cm high box',
                     'sitting on a chair', 'climbing down stairs', 'climb a ladder',
                     'leaning against a wall', 'dips on a box', 'vault over an obstacle',
                     'walks using crutches', 'sits on a sunbed', 'pushes a large object']:
            with self.subTest(text=text):
                self.assertTrue(classify({'description':text}))

    def test_preserve_flat_ground_and_word_collisions(self):
        for text in ['boxing and punching', 'dancing hip hop box step',
                     'jumps in one place and lands on the ground', 'sitting on the floor',
                     'stands on one leg', 'picks up a box', 'overturns a chair',
                     'hiding behind a wall', 'looks over a fence', 'takes an item from a table',
                     'jumps over a small fence to the right']:
            with self.subTest(text=text):
                self.assertFalse(classify({'description':text}))

    def test_support_context_across_descriptions(self):
        self.assertTrue(classify({'short':'crouching while supporting with hand',
                                  'technical':'hiding behind wall'}))

    def test_source_name_overrides_conflicting_dance_description(self):
        self.assertTrue(classify({'source_motion_name':'come_down_50cm_box_R_004',
                                  'description':'a sequence of simple dance steps'}))


if __name__ == '__main__':
    unittest.main()
