#!/usr/bin/env python

import numpy as np
import scipy.sparse
import time
from codecs import open
from collections import defaultdict

def run(text_path='text8', vocab_path='vocab.txt', output_path='vectors.txt'):
    begin = time.time()
    with open(text_path, "r", "utf-8") as f:
        text = f.readlines()

    with open(vocab_path, "r", "utf-8") as v:
        vocab = v.readlines()

    vocab1 = [word.strip() for word in vocab]
    vocab = lambda: list(vocab1)
    vocabulary = defaultdict(vocab)

    corpus = []
    token_ids = []
    for line in text:
        words = line.split()
        corpus += [words[i] for i in range(len(words) - 1)]

    for i, line in enumerate(corpus):
        tok = line.strip().split()
        token_ids = [vocabulary[word][0] for word in tok]

    cooccurr = make_cooccurMatrix(token_ids, vocabulary)
    alpha = 0.01
    epochs = 25
    omatrix = regularGlove(vocab1, cooccurr, alpha, epochs)
    omatrix = np.around(omatrix, 3)
    vocab = np.array(vocab1)
    strings = [vocab[i] + ' ' + ' '.join([str(x) for x in omatrix[i]]) for i in range(len(vocab))]
    text = '\n'.join(strings)
    with open(output_path, "w") as text_file:
        text_file.write(text)

    print("Time taken " + (str)(time.time() - begin))

def make_cooccurMatrix(token_ids, vocab):
    windowsize = 5
    cooccurMatrix = scipy.sparse.lil_matrix((len(vocab), len(vocab)))

    for target, target_id in enumerate(token_ids):
        context_ids = token_ids[max(0, target - windowsize): target]

        for left_i, left_id in enumerate(context_ids):
            distance = len(context_ids) - left_i
            weight = 1.0 / float(distance)
            cooccurMatrix[target_id, left_id] += weight     # Symmetry
            cooccurMatrix[left_id, target_id] += weight

    for i, (row, data) in enumerate(zip(cooccurMatrix.rows,cooccurMatrix.data)):
        for data_idx, j in enumerate(row):
            yield i, j, data[data_idx]

def regularGlove(vocab, cooccurrences, alpha, epochs):
    data = []
    wordVectors = np.random.uniform(low=-0.5, high=0.5, size=(len(vocab)+len(vocab), 300))
    biases = np.random.uniform(low=-0.5, high=0.5, size=(len(vocab) + len(vocab), ))
    grad_sq = np.ones((len(vocab)+len(vocab), 300))
    grad_sqB = np.ones(len(vocab)+len(vocab))

    for i_target, i_context, Xij in cooccurrences:
        data = [(wordVectors[i_target], wordVectors[i_context + len(vocab)], biases[i_target: i_target + 1],
             biases[i_context + len(vocab): i_context + len(vocab) + 1], grad_sq[i_target], grad_sq[i_context + len(vocab)],
             grad_sqB[i_target: i_target + 1], grad_sqB[i_context + len(vocab): i_context + len(vocab) + 1], Xij)]

    Xmax = 100
    for j in range(epochs):
        np.random.shuffle(data)

        for (vTarget, vContext, target_bias, context_bias, target_gradsq, context_gradsq, target_gradsqB, context_gradsqB, Xij) in data:

            weight = np.power((Xij / Xmax), 0.75) if Xij < Xmax else 1
            innerbracketJ = (vTarget.dot(vContext) + target_bias[0] + context_bias[0] - np.log(Xij))

            vTarget -= (alpha * innerbracketJ * vContext / np.sqrt(target_gradsq)) #coz target gradient = innerbracketJ *vContext
            vContext -= (alpha * innerbracketJ * vTarget / np.sqrt(context_gradsq)) #coz context gradient = innerbracketJ *vTargwt
            target_bias -= (alpha * innerbracketJ / np.sqrt(target_gradsqB)) #coz gradient for bias term for target word = innerbracketJ
            context_bias -= (alpha * innerbracketJ / np.sqrt(context_gradsqB)) #coz gradient for bias term for context word = innerbracketJ

            target_gradsq += np.square(innerbracketJ * vContext)
            context_gradsq += np.square(innerbracketJ * vTarget)
            target_gradsqB += np.square(innerbracketJ) #coz gradient for bias term for target word = innerbracketJ
            context_gradsqB += np.square(innerbracketJ) #coz gradient for bias term for context word = innerbracketJ
    return wordVectors
run()
