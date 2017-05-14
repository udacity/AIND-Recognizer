import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        return None

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model = None
        bic = float("-inf")
        model_list = []
        for n_components in range(self.min_n_components, self.max_n_components):
            try:
                # Train
                model = GaussianHMM(n_components=n_components,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
                # Test
                logL = model.score(self.X, self.lengths)
                parameters = n_components * n_components + 2 * n_components * len(self.X[0]) - 1
                bic = -2 * logL + parameters * np.log(len(self.X))
            except Exception as e:  # TODO: Find out Why Sometimes Exceptions Occur
                continue
            model_list.append([bic, model])
        model_sorted = sorted(model_list, key=lambda m: m[0])
        result_model = None
        try:
            result_model = model_sorted[0][1]
        except Exception as e:
            pass
        return result_model  # Logs are Negative, to Maximize Choose the Last (Should Probably Just Add Parameters to the Sort Function)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def get_other_words_likelihood(self, model):
        likelihood = 0
        m = len(self.hwords) - 1
        for word, xlength in self.hwords.items():
            if word is not self.this_word:
                likelihood += model.score(xlength[0], xlength[1])
        likelihood /= m
        return likelihood

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model = None
        bic = float("-inf")
        model_list = []
        for n_components in range(self.min_n_components, self.max_n_components):
            try:
                # Train
                model = GaussianHMM(n_components=n_components,
                                    covariance_type="diag",
                                    n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
                # Test
                logL = model.score(self.X, self.lengths)
                other_words_likelihood = self.get_other_words_likelihood(model)
                dic = logL - other_words_likelihood
            except Exception as e:  # TODO: Find out Why Sometimes Exceptions Occur
                continue
            model_list.append([dic, model])
        model_sorted = sorted(model_list, key=lambda m: m[0])
        result_model = None
        try:
            result_model = model_sorted[-1][1]
        except Exception as e:
            pass
        return result_model  # Logs are Negative, to Maximize Choose the Last (Should Probably Just Add Parameters to the Sort Function)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds'''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Break Data Sets into Folds
        word_sequences = self.sequences  # get sequences to split them in folds
        len_word_sequences = len(word_sequences)
        # Make Sure Data Can be Safely Broken
        if len_word_sequences < 3:
            best_num_components = self.n_constant
            return self.base_model(best_num_components)
        else:
            split_method = KFold()
        # Model Selection
        model = None
        model_list = []
        for n_components in range(self.min_n_components, self.max_n_components):
            log_list = []
            try:
                for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                    # Train
                    X, lengths = combine_sequences(cv_train_idx, word_sequences)
                    model = GaussianHMM(n_components=n_components,
                                        covariance_type="diag",
                                        n_iter=1000,
                                        random_state=self.random_state,
                                        verbose=False).fit(X, lengths)

                    # Test
                    X, lengths = combine_sequences(cv_test_idx, word_sequences)
                    logL = model.score(X, lengths)
                    log_list.append(logL)
            except Exception as e:  # TODO: Find out Why Sometimes Exceptions Occur
                continue
            mean = np.mean(log_list)
            model_list.append([mean, model])
        model_sorted = sorted(model_list, key=lambda m: m[0])
        result_model = None
        try:
            result_model = model_sorted[-1][1]
        except Exception as e:
            pass
        return result_model  # Logs are Negative, to Maximize Choose the Last (Should Probably Just Add Parameters to the Sort Function)
