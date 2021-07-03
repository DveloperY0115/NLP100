import numpy as np
import pandas as pd
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

import torch
from torch.utils.data import Dataset

# TODO: Replace preprocessing with torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab


class NewsAggregatorDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        """
        Get the number of data
        """
        assert (len(self.X) == len(self.y))
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get the data at the given index
        """
        return self.X[idx], self.y[idx]


def tokenize_and_label(df, stop_words=None, one_hot_labels=None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    if one_hot_labels is None:
        one_hot_labels = {'b': 0, 't': 1, 'm': 2, 'e': 3}

    tokenized = []
    labels = []

    """
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    
    for sentence in df['title']:
        counter.update(tokenizer(sentence))
    vocab = Vocab(counter, min_freq=1)
    """

    for category in df['category']:
        label = [0] * len(one_hot_labels.keys())
        label[one_hot_labels[category]] = 1
        labels.append(label)

    return tokenized, labels


def encode_words(tokenized, word_to_index):
    encoded = []
    for line in tokenized:
        temp = []
        for word in line:
            try:
                temp.append(word_to_index[word])
            except KeyError:
                temp.append(word_to_index['unk'])

        encoded.append(temp)

    return encoded


def add_padding(encoded, word_to_index):
    result = encoded
    max_len = max(len(l) for l in result)

    for line in result:
        if len(line) < max_len:
            line += [word_to_index['pad']] * (max_len - len(line))

    return result


def process_data(save=False, out_dir='./data'):
    stop_words = set(stopwords.words('english'))

    # read data files
    train_data = pd.read_csv('./data/train.csv', sep='\t')
    valid_data = pd.read_csv('./data/valid.csv', sep='\t')
    test_data = pd.read_csv('./data/test.csv', sep='\t')

    # process train data
    train_tokenized, train_label = tokenize_and_label(train_data)

    # process validation data
    valid_tokenized, valid_label = tokenize_and_label(valid_data)

    # process test data
    test_tokenized, test_label = tokenize_and_label(test_data)

    # create vocab
    vocab = FreqDist(np.hstack(train_tokenized))
    vocab_size = 1000
    vocab = vocab.most_common(vocab_size)

    # word to index
    word_to_index = {word[0]: index + 2 for index, word in enumerate(vocab)}
    word_to_index['pad'] = 1
    word_to_index['unk'] = 0

    # encode data according to dictionary 'word_to_index'

    train_encoded = encode_words(train_tokenized, word_to_index)

    # encode validation set
    valid_encoded = encode_words(valid_tokenized, word_to_index)

    # encode test set
    test_encoded = encode_words(test_tokenized, word_to_index)

    # add padding
    max_lens = [max(len(l) for l in train_encoded), max(len(l) for l in valid_encoded), max(len(l) for l in test_encoded)]

    max_len = max(max_lens)

    for line in train_encoded:
        if len(line) < max_len:
            line += [word_to_index['pad']] * (max_len - len(line))

    for line in valid_encoded:
        if len(line) < max_len:
            line += [word_to_index['pad']] * (max_len - len(line))

    for line in test_encoded:
        if len(line) < max_len:
            line += [word_to_index['pad']] * (max_len - len(line))

    if save:
        pass

    return train_encoded, train_label, valid_encoded, valid_label, test_encoded, test_label


def create_dataset(X, y):
    """
    Create Pytorch dataset
    """
    return NewsAggregatorDataset(X, y)
