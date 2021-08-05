"""
preprocessing.py - Collection of functions used for preprocessing in NLP projects.
"""

import numpy as np
import nltk

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

import string
import spacy

spacy_en = spacy.load("en")


def cast_lower(sentence):
    """
    Convert all characters in a sentence into lower case.
    """
    return sentence.str.lower()


def remove_punctuation(sentence):
    """
    Remove punctuation in a sentence.
    """
    PUNCT_TO_REMOVE = string.punctuation
    return sentence.translate(str.maketrans("", "", PUNCT_TO_REMOVE))


def remove_stopwords(sentence):
    """
    Remove stopwords from a sentence.
    """
    STOPWORDS = set(stopwords.words("english"))
    return " ".join([word for word in str(sentence).split() if word not in STOPWORDS])


def lemmatize_words(sentence):
    """
    Lemmatize the given sentence
    """
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

    pos_tagged_text = nltk.pos_tag(sentence.split())

    return " ".join(
        [
            lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
            for word, pos in pos_tagged_text
        ]
    )


def tokenize_sentence(sentence):
    """
    Tokenize the given sentence.
    """
    return [tok.text for tok in spacy_en.tokenizer(sentence)]


def build_vocab(corpus, vocab_size):
    """
    Build vocabularly from the given corpus (i.e. set of sentences).
    """
    vocab = FreqDist(np.hstack(corpus)).most_common(vocab_size)
    word_to_index = {word[0]: index + 2 for index, word in enumerate(vocab)}
    word_to_index["pad"] = 1
    word_to_index["unk"] = 0
    return vocab, word_to_index
