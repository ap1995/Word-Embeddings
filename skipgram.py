import numpy as np
import random
import time
import itertools
import concurrent.futures
import multiprocessing
from codecs import open
from collections import Counter

def main():
    text8_data_file = 'text8'
    vocab_data = 'vocab.txt'
    with open(text8_data_file, "r", "utf-8") as f:
        text = f.readlines()

    with open(vocab_data, "r", "utf-8") as v:
        vocab = v.readlines()

    vocabulary = [word.strip() for word in vocab]
    print("Vocab size " + str(len(vocab)))
    # create_one_hot(vocabulary)

    hmatrix = np.random.random((len(vocabulary), 300))
    omatrix = np.random.random((300, len(vocabulary)))
    tuples = []
    for line in text:
        words = line.split()
        tuples += [(words[i], words[i + 1]) for i in range(len(words) - 1)]

    small_tuples = tuples[0:100]
    start_time = time.time()
    myTuples = []
    alpha = 0.01
    for i in range(0,len(small_tuples)):
        if(small_tuples[i][0] in vocabulary and (small_tuples[i][1] in vocabulary)):
            myTuples += (small_tuples[i][0], small_tuples[i][1])

            in_indx = vocabulary.index(tuples[i][0])
            out_indx = vocabulary.index(tuples[i][1])
            h = hmatrix[in_indx]
            # print("Hshape")
            print(h[:10])
            # print("Omatrix")
            # print("test shape")
            test = omatrix[:,out_indx]
            print(test[:10])
            ovector = h.dot(test[:,None])
            print("Ovector")
            print(ovector)
            # print(ovector[:out_indx])
            output = sigmoid(ovector)
            print("Output")
            print(output)
            actual = np.zeros(len(vocabulary))
            print("Actual")
            print(actual.shape)
            actual[out_indx] = 1
            error = output - actual[out_indx]
            print("Error:")
            print(error)
            omatrix[:, in_indx] -= alpha*error*h
            EH = error*omatrix[:, in_indx]
            hmatrix[in_indx] -= alpha * EH.transpose()
        else:
            continue

    print("Time taken "+ (str)(time.time()-start_time))

    # print(myTuples)
    # print(len(myTuples))
    # weight_matrix1(vocabulary)

# made one hot vectors for all words in the vocab, put together
# def create_one_hot(vocabulary):
#     wordvecs = np.zeros((len(vocabulary), len(vocabulary)), int)
#     for i, word in enumerate(vocabulary):
#         wordvecs[i, vocabulary.index(word)] = 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

main()
