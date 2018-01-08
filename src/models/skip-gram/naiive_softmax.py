# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter

# flatten 2-D iterable to 1-D
flatten = lambda lists: [item for sublist in lists for item in sublist]
random.seed(1024)

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    if eindex >= len(train_data):
        batch = train_data[sindex: ]
        yield batch


def prepare_sequence(seq, word2index):
    """get the word index of the sequence"""
    idxs = list(map(lambda w: word2index[w] if word2index[w] is not None else word2index["<UNK>"], seq))
    return Variable(LongTensor)


def prepare_word(word, word2index):
    return Variable(LongTensor([word2index[word]] if word2index.get(word) is not None
                               else LongTensor([word2index["<UNK>"]])))


print(nltk.corpus.gutenberg.fileids())
# sampling sentences for test
corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:100]
corpus = [[word.lower() for word in sent] for sent in corpus]
print(corpus)

# extract Stopwords from unigram distribution's tails
word_count = Counter(flatten(corpus))
print("word count: ", word_count)
border = int(len(word_count) * 0.01)
stopwords = word_count.most_common()[:border] + list(reversed(word_count.most_common()))[:border]
stopwords = [s[0] for s in stopwords]
print(stopwords)

# build vocabulary
vocab = list(set(flatten(corpus)) - set(stopwords))
vocab.append('<UNK>')
print("the vocab is %d, the corpus is %d" % (len(vocab), len(flatten(corpus))))
word2index = {'<UNK>': 0}
for v in vocab:
    if word2index.get(v) is None:
        word2index[v] = len(word2index)

index2word = {v: k for k, v in word2index.items()}

print(word2index)

# prepare train data
WINDOW_SIZE = 3
windows = flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE +
                                    c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])
print(windows[0])
print(windows[1])
print(windows[2])

train_data = []
for window in windows:
    for i in range(WINDOW_SIZE * 2 + 1):
        if i == WINDOW_SIZE or window[i] == '<DUMMY>':
            continue
        train_data.append((window[WINDOW_SIZE], window[i]))
print("the train_data is:", train_data[: WINDOW_SIZE * 2])

X_p = []
Y_p = []
for tr in train_data:
    X_p.append(prepare_word(tr[0], word2index).view(1, -1))
    Y_p.append(prepare_word(tr[1], word2index).view(1, -1))

train_data = list(zip(X_p, Y_p))
print("the data size is :", len(train_data))   # 7606


