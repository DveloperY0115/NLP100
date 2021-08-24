"""
dataset.py - Classes & functions related to dataset
"""

import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import nltk

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
import spacy

spacy_en = spacy.load("en")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


class AggregatorNewsDataset(data.Dataset):
    def __init__(self, data_file, num_freq_word=20, num_rare_word=20, vocab_size=5000):
        super(AggregatorNewsDataset, self).__init__()

        # read CSV file
        self.df = pd.read_csv(data_file)
        self.df["title"] = self.df["title"].astype(str)

        # lower casting
        self.df["title_processed"] = self.df["title"].str.lower()

        # removal of punctuations
        PUNCT_TO_REMOVE = string.punctuation

        def remove_punctuation(text):
            return text.translate(str.maketrans("", "", PUNCT_TO_REMOVE))

        self.df["title_processed"] = self.df["title_processed"].apply(
            lambda text: remove_punctuation(text)
        )

        # removal of stopwords
        STOPWORDS = set(stopwords.words("english"))

        def remove_stopwords(text):
            return " ".join([word for word in str(text).split() if word not in STOPWORDS])

        self.df["title_processed"] = self.df["title_processed"].apply(
            lambda text: remove_stopwords(text)
        )

        # removal of frequent words
        self.counter = Counter()
        for text in self.df["title_processed"].values:
            for word in text.split():
                self.counter[word] += 1

        FREQWORDS = set([w for (w, wc) in self.counter.most_common(num_freq_word)])

        def remove_freqwords(text):
            return " ".join([word for word in str(text).split() if word not in FREQWORDS])

        self.df["title_processed"] = self.df["title_processed"].apply(
            lambda text: remove_freqwords(text)
        )

        # removal of rare words
        RAREWORDS = set([w for (w, wc) in self.counter.most_common()[: -num_rare_word - 1 : -1]])

        def remove_rarewords(text):
            return " ".join([word for word in str(text).split() if word not in RAREWORDS])

        self.df["title_processed"] = self.df["title_processed"].apply(
            lambda text: remove_rarewords(text)
        )

        lemmatizer = WordNetLemmatizer()
        wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

        def lemmatize_words(text):
            pos_tagged_text = nltk.pos_tag(text.split())
            return " ".join(
                [
                    lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN))
                    for word, pos in pos_tagged_text
                ]
            )

        self.df["title_processed"] = self.df["title_processed"].apply(
            lambda text: lemmatize_words(text)
        )

        # tokenize
        def tokenize(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        self.df["title_processed"] = self.df["title_processed"].apply(lambda text: tokenize(text))

        # build vocab
        token = []

        for idx, row in self.df.iterrows():
            token.append(row["title_processed"])

        vocab = FreqDist(np.hstack(token))

        self.vocab = vocab.most_common(vocab_size)

        self.word_to_index = {word[0]: index + 2 for index, word in enumerate(self.vocab)}
        self.word_to_index["pad"] = 1
        self.word_to_index["unk"] = 0

        self.vocab_size = len(self.word_to_index)

        # encode tokens into integers
        def encode_text(tokenized_text):
            encoded = []

            for word in tokenized_text:
                try:
                    encoded.append(self.word_to_index[word])
                except KeyError:
                    encoded.append(self.word_to_index["unk"])

            return encoded

        self.df["title_processed"] = self.df["title_processed"].apply(
            lambda text: encode_text(text)
        )

        # add padding
        seq_max_len = 0

        for idx in range(len(self.df)):
            if len(self.df["title_processed"][idx]) > seq_max_len:
                seq_max_len = len(self.df["title_processed"][idx])

        for idx in range(len(self.df)):
            seq = self.df["title_processed"][idx]

            for _ in range(seq_max_len - len(seq)):
                seq.append(0)

        publisher_one_hot = pd.get_dummies(self.df["publisher"])
        category_one_hot = pd.get_dummies(self.df["category"])

        self.df = self.df.drop("title", axis=1)
        self.df.drop("publisher", axis=1)
        self.df.drop("category", axis=1)

        # convert to torch.Tensor
        self.title = torch.tensor(np.array(self.df["title_processed"].tolist()))
        self.publisher = torch.tensor(publisher_one_hot.to_numpy())
        self.category = torch.tensor(category_one_hot.to_numpy())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.title[index], self.publisher[index], self.category[index]
