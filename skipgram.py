#!/usr/bin/env python

import numpy as np
import random
import time
from codecs import open

def run(text_path='text8', vocab_path='vocab.txt', output_path='vectors.txt'):
    begin = time.time()
    with open(text_path, "r", "utf-8") as f:
        text = f.readlines()

    with open(vocab_path, "r", "utf-8") as v:
        vocab = v.readlines()

    vocab = [word.strip() for word in vocab]
    vocabulary = set(vocab)
    print("Vocab size " + str(len(vocab)))

    hmatrix = np.random.random((len(vocabulary), 300))

    tuples = []
    corpus = []
    for line in text:
        words = line.split()
        corpus += [words[i] for i in range(len(words) - 1)]
        tuples += [(words[i], words[i + 1]) for i in range(len(words) - 1)]

    freqTable = wordFrequency(corpus, vocab, vocabulary)
    negTable = create_negativeTable(freqTable, vocab)
    small_tuples = tuples
    alpha = 0.015
    epochs = 10
    for i in range(epochs):
        hmatrix = regularSG(small_tuples, vocab, vocabulary, hmatrix, alpha, negTable)
        alpha = alpha / 2
    hmatrix = np.around(hmatrix, 3)
    vocab = np.array(vocab)
    strings = [vocab[i] + ' ' + ' '.join([str(x) for x in hmatrix[i]]) for i in range(len(vocab))]
    text = '\n'.join(strings)
    with open(output_path, "w") as text_file:
        text_file.write(text)

    print("Time taken " + (str)(time.time() - begin))

def sigmoid(x):
    return 1 / float(1 + np.exp(-x))

def wordFrequency(corpus, vocab, vocabulary):
    freqTable = np.zeros(len(vocab))
    for word in corpus:
        if (word in vocabulary):
            freqTable[vocab.index(word)] += 1
    return freqTable

def create_negativeTable(freqTable, vocab):
    negTable = []
    power = 0.75
    freqInNTable = np.power(freqTable, power)
    denom = freqInNTable.sum()
    freqInNTable = (freqInNTable *100000000) / denom
    for i in range(0, len(vocab)):
        negTable.extend([vocab[i]]* int(freqInNTable[i]))
    return negTable

def regularSG(small_tuples, vocab, vocabulary, hmatrix, alpha, negTable):
    omatrix = np.zeros((300, len(vocabulary)))
    for i in range(0, len(small_tuples)):
        if (small_tuples[i][0] in vocabulary and (small_tuples[i][1] in vocabulary)):
            in_indx = vocab.index(small_tuples[i][0])
            out_indx = vocab.index(small_tuples[i][1])
            h = hmatrix[in_indx]
            test = omatrix[:, out_indx]
            ovector = h.dot(test)
            output = sigmoid(ovector)
            actual = 1
            error = output - actual
            omatrix[:, out_indx] -= alpha * error * h
            EH = error * omatrix[:, out_indx]
            hmatrix[in_indx] -= alpha * EH.transpose()
            # Negative Sampling Updates
            negSamples = random.sample(negTable, 5)
            for i in range(0,5):
                n_out_indx = vocab.index(negSamples[i])
                ntest = omatrix[:, n_out_indx]
                n_ovector = h.dot(ntest)
                n_output = sigmoid(n_ovector)
                # n_actual =0
                omatrix[:, n_out_indx] -= alpha * n_output * h #n_error = n_output coz n_actual=0
                n_EH = n_output * omatrix[:, n_out_indx]
                hmatrix[in_indx] -= alpha * n_EH.transpose()
        else:
            continue

    return hmatrix
run()
