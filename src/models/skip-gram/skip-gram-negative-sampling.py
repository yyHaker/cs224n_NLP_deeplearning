# -*- coding: utf-8 -*-
"""
the toy realization of word2vec skip-gram model
1. negative sampling
reference:
[1] Skip-gram with negative sampling: https://nbviewer.jupyter.org
/github/DSKSD/DeepNLP-models-Pytorch/blob/master/notebooks
/02.Skip-gram-Negative-Sampling.ipynb
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter
flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(1024)

USE_CUDA = torch.cuda.is_available()
# gpus = [0]
# torch.cuda.set_device(gpus[0])

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
        batch = train_data[sindex:]
        yield batch


def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return Variable(LongTensor(idxs))


def prepare_word(word, word2index):
    return Variable(LongTensor([word2index[word]]) if word2index.get(word) is not None
                    else LongTensor([word2index["<UNK>"]]))


# data load and preprocessing
corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:500]
corpus = [[word.lower() for word in sent] for sent in corpus]

# Exclude sparse words
word_count = Counter(flatten(corpus))
MIN_COUNT = 3
exclude = []
for w, c in word_count.items():
    if c < MIN_COUNT:
        exclude.append(w)

# prepare the train data
vocab = list(set(flatten(corpus)) - set(exclude))
word2index = {}
for vo in vocab:
    if word2index.get(vo) is None:
        word2index[vo] = len(word2index)

index2word = {v: k for k, v in word2index.items()}

WINDOW_SIZE = 5
windows = flatten(
    [list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])

train_data = []
# prepare train word pairs
for window in windows:
    for i in range(WINDOW_SIZE * 2 + 1):
        if window[i] in exclude or window[WINDOW_SIZE] in exclude:
            continue  # min_count
        if i == WINDOW_SIZE or window[i] == '<DUMMY>':
            continue
        train_data.append((window[WINDOW_SIZE], window[i]))

X_p = []
y_p = []

for tr in train_data:
    X_p.append(prepare_word(tr[0], word2index).view(1, -1))
    y_p.append(prepare_word(tr[1], word2index).view(1, -1))

train_data = list(zip(X_p, y_p))
print("the train data size is", len(train_data))

# Build Unigram Distribution**0.75 P(w)=U(w)3/4 /Z
Z = 0.001
word_count = Counter(flatten(corpus))
num_total_words = sum([c for w, c in word_count.items() if w not in exclude])

unigram_table = []
for vo in vocab:
    unigram_table.extend([vo] * int(((word_count[vo]/num_total_words)**0.75)/Z))
print("the vocabulary size is %d, the unigram table size is %d" % (len(vocab), len(unigram_table)))
print("sample the unigram table...")
print(unigram_table[:100])
print("-"*50)


# Negative Sampling
def negative_sampling(targets, unigram_table, k):
    """
    :param targets: target words, shape(batch, 1)
    :param unigram_table:
    :param k: the num of sampling
    :return: [batch, k]
    """
    batch_size = targets.size(0)
    neg_samples = []
    for i in range(batch_size):
        nsample = []
        target_index = targets[i].data.cpu().tolist()[0] if USE_CUDA else targets[i].data.tolist()[0]
        while len(nsample) < k:  # num of sampling
            neg = random.choice(unigram_table)
            if word2index[neg] == target_index:
                continue
            nsample.append(neg)
        neg_samples.append(prepare_sequence(nsample, word2index).view(1, -1))  # a list of 1xk
    return torch.cat(neg_samples)    # batch x k


# models
class SkipgramNegSampling(nn.Module):
    def __init__(self, vocab_size, projection_dim):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)  # center embedding
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)  # out embedding
        self.logsigmoid = nn.LogSigmoid()

        initrange = (2.0 / (vocab_size + projection_dim)) ** 0.5  # Xavier init
        self.embedding_v.weight.data.uniform_(-initrange, initrange)  # init
        self.embedding_u.weight.data.uniform_(-0.0, 0.0)  # init

    def forward(self, center_words, target_words, negative_words):
        center_embeds = self.embedding_v(center_words)  # B x 1 x D
        target_embeds = self.embedding_u(target_words)  # B x 1 x D

        neg_embeds = -self.embedding_u(negative_words)  # B x K x D

        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)  # Bx1
        negative_score = torch.sum(neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2), 1).\
            view(neg_embeds.size(0), -1)  # BxKxD * BxDx1 -> BxKx1 -> BxK -> B-> Bx1

        loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)  # B x 1
        # TODO: how to realize the negative sampling?
        return -torch.mean(loss)

    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)

        return embeds

EMBEDDING_SIZE = 30
BATCH_SIZE = 256
EPOCH = 100
NEG = 10  # Num of Negative Sampling

losses = []
model = SkipgramNegSampling(len(word2index), EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCH):
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        inputs, targets = zip(*batch)

        inputs = torch.cat(inputs)  # B x 1
        targets = torch.cat(targets)  # B x 1
        negs = negative_sampling(targets, unigram_table, NEG)
        model.zero_grad()

        loss = model(inputs, targets, negs)

        loss.backward()
        optimizer.step()

        losses.append(loss.data.tolist()[0])
    if epoch % 10 == 0:
        print("Epoch : %d, mean_loss : %.02f" % (epoch, np.mean(losses)))
        losses = []


def word_similarity(target, vocab):
    """
    :param target:  the target word
    :param vocab: the vocab
    :return:
    """
    target_V = model.prediction(prepare_word(target, word2index))
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target:
            continue
        vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        cosine_sim = F.cosine_similarity(target_V, vector).data.tolist()[0]
        similarities.append([vocab[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]


# test the word similarity
for i in range(5):
    test = random.choice(list(vocab))
    print("test the similarity of word ", test)
    print("-"*50)
    print(word_similarity(test, vocab))
    print("-"*50)

