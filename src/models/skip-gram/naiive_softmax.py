# -*- coding:utf-8 -*-
"""
a toy realization of word2vec
1. statistics of training word pairs and cut off stopwords
2. batch training
3. one-hidden layer and a softmax classifier
4. negative log-likelihood

reference:
[1]Word2Vec Tutorial - The Skip-Gram Model:
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
[2]Skip-gram with naiive softmax: https://nbviewer.jupyter.org/github/DSKSD/
DeepNLP-models-Pytorch/blob/master/notebooks/01.Skip-gram-Naive-
Softmax.ipynb
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

# flatten 2-D iterable to 1-D
flatten = lambda lists: [item for sublist in lists for item in sublist]
random.seed(1024)

USE_CUDA = torch.cuda.is_available()
gpus = [0]
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
        batch = train_data[sindex: ]
        yield batch


def prepare_sequence(seq, word2index):
    """get the word index of the sequence"""
    idxs = list(map(lambda w: word2index[w] if word2index[w] is not None else word2index["<UNK>"], seq))
    return Variable(LongTensor(idxs))


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
print("the sample train_data is:", train_data[: WINDOW_SIZE * 2])

X_p = []
Y_p = []
for tr in train_data:
    X_p.append(prepare_word(tr[0], word2index).view(1, -1))
    Y_p.append(prepare_word(tr[1], word2index).view(1, -1))

train_data = list(zip(X_p, Y_p))
print("the data size is :", len(train_data))   # 7606


# the skip-gram model
class Skipgram(nn.Module):
    def __init__(self, vocab_size, projection_dim):
        super(Skipgram, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)  # 即Word Embeddings
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)

        self.embedding_v.weight.data.uniform_(-1, 1)  # init
        self.embedding_u.weight.data.uniform_(0, 0)  # init
        # self.out = nn.Linear(projection_dim,vocab_size)

    def forward(self, center_words, target_words, outer_words):
        """
        :param center_words: the center word, batch
        :param target_words: the target word, batch
        :param outer_words:
        :return:
        """
        center_embeds = self.embedding_v(center_words)  # B x 1 x D
        target_embeds = self.embedding_u(target_words)  # B x 1 x D
        outer_embeds = self.embedding_u(outer_words)  # B x V x D

        # Performs a batch matrix-matrix product of matrices
        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)  # Bx1xD * BxDx1 => Bx1
        norm_scores = outer_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)  # BxVxD * BxDx1 => BxV

        nll = -torch.mean(
            torch.log(torch.exp(scores) / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1)))  # log-softmax

        return nll  # negative log likelihood

    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)

        return embeds


EMBEDDING_SIZE = 100
BATCH_SIZE = 256
EPOCH = 100

losses = []
model = Skipgram(len(word2index), EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(EPOCH):
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        inputs, targets = zip(*batch)

        inputs = torch.cat(inputs)  # Bx1
        targets = torch.cat(targets)  # Bx1
        vocabs = prepare_sequence(list(vocab), word2index).expand(inputs.size(0), len(vocab))  # B x V

        model.zero_grad()  # Sets gradients of all model parameters to zero.
        loss = model(inputs, targets, vocabs)

        loss.backward()   # backpropagation
        optimizer.step()  # Performs a single optimization step

        losses.append(loss.data.tolist()[0])

    if epoch % 10 == 0:
        print("Epoch:  %d, mean loss : %.02f" % (epoch, np.mean(losses)))
        losses = []


def word_similarity(target, vocab):
    """
    :param target: the target word
    :param vocab: the vocabulary
    :return:
    """
    target_v = model.prediction(prepare_word(target, word2index))
    similarities = []
    for i in range(len(vocab)):
        if vocab[i] == target:
            continue
        vector = model.prediction(prepare_word(list(vocab)[i], word2index))
        cosine_sim = F.cosine_similarity(target_v, vector).data.tolist()[0]
        similarities.append([vocab[i], cosine_sim])
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]  # sort by similarity


# test the word similarity
for i in range(5):
    test = random.choice(list(vocab))
    print("test the similarity of word ", test)
    print("-"*50)
    print(word_similarity(test, vocab))
    print("-"*50)




