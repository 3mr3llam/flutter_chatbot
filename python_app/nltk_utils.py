import nltk
# nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# tokenize each word in the sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# get the stem for the word and lower it
def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    # loop over all words
    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag


