from unittest import TestCase

from asl_data import AslDb
from my_model_selectors import (
    SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV,
)

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
