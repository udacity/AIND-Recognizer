from unittest import TestCase

from asl_data import AslDb
from asl_utils import train_all_words
from my_model_selectors import (
    SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV,
)
from my_recognizer import recognize

""" DEPRECATED MODULE
This module has been split into two new modules: asl_test_model_selectors.py and asl_test_recognizer.py
This module is included in the repo for the sake of legacy code that still uses it.
"""
FEATURES = ['right-y', 'right-x']


class TestSelectors(TestCase):
    def setUp(self):
        asl = AslDb()
        self.training = asl.build_training(FEATURES)
        self.sequences = self.training.get_all_sequences()
        self.xlengths = self.training.get_all_Xlengths()

    def test_select_constant_interface(self):
        model = SelectorConstant(self.sequences, self.xlengths, 'BUY').select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorConstant(self.sequences, self.xlengths, 'BOOK').select()
        self.assertGreaterEqual(model.n_components, 2)

    def test_select_bic_interface(self):
        model = SelectorBIC(self.sequences, self.xlengths, 'FRANK').select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorBIC(self.sequences, self.xlengths, 'VEGETABLE').select()
        self.assertGreaterEqual(model.n_components, 2)

    def test_select_cv_interface(self):
        model = SelectorCV(self.sequences, self.xlengths, 'JOHN').select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorCV(self.sequences, self.xlengths, 'CHICKEN').select()
        self.assertGreaterEqual(model.n_components, 2)

    def test_select_dic_interface(self):
        model = SelectorDIC(self.sequences, self.xlengths, 'MARY').select()
        self.assertGreaterEqual(model.n_components, 2)
        model = SelectorDIC(self.sequences, self.xlengths, 'TOY').select()
        self.assertGreaterEqual(model.n_components, 2)


class TestRecognize(TestCase):
    def setUp(self):
        self.asl = AslDb()
        self.training_set = self.asl.build_training(FEATURES)
        self.test_set = self.asl.build_test(FEATURES)
        self.models = train_all_words(self.training_set, SelectorConstant)

    def test_recognize_probabilities_interface(self):
        probs, _ = recognize(self.models, self.test_set)
        self.assertEqual(len(probs), self.test_set.num_items, "Number of test items in probabilities list incorrect.")
        self.assertEqual(len(probs[0]), self.training_set.num_items,
                         "Number of training word probabilities in test item dictionary incorrect.")
        self.assertEqual(len(probs[-1]), self.training_set.num_items,
                         "Number of training word probabilities in test item dictionary incorrect.")
        self.assertIn('FRANK', probs[0], "Dictionary of probabilities does not contain correct keys")
        self.assertIn('CHICKEN', probs[-1], "Dictionary of probabilities does not contain correct keys")

    def test_recognize_guesses_interface(self):
        _, guesses = recognize(self.models, self.test_set)
        self.assertEqual(len(guesses), self.test_set.num_items, "Number of test items in guesses list incorrect.")
        self.assertIsInstance(guesses[0], str, "The guesses are not strings")
        self.assertIsInstance(guesses[-1], str, "The guesses are not strings")
