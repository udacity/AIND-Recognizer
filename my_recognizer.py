import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    xlengths = test_set.get_all_Xlengths()
    for word_idx, xlength in xlengths.items():
        word_dict = {}
        current_guess = [float("-inf"), ""]
        for word, model in models.items():
            logL = float("-inf")
            try:
                logL = model.score(xlength[0], xlength[1])
                if current_guess[0] < logL:  # Check if a This Word Has Better Likelihood
                    current_guess[0] = logL
                    current_guess[1] = word
            except Exception as e:
                pass
            word_dict[word] = logL
        probabilities.append(word_dict)
        guesses.append(current_guess[1])
    return probabilities, guesses
