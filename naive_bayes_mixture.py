# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as numpy
import math
from collections import Counter





def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """



    # TODO: Write your code here
    positiveCounter = Counter()
    negativeCounter = Counter()
    for i in range(0, len(train_set)):
        for word in train_set[i]:
            if train_labels[i] == 1: #positive
                positiveCounter[word] +=1
            if train_labels[i] == 0: #negative
                negativeCounter[word] +=1

    num_total_words_negative = sum(negativeCounter.values())
    num_uniquewords_negative = len(list(negativeCounter))
    num_total_words_positive = sum(positiveCounter.values())
    num_uniquewords_positive = len(list(positiveCounter))

    positiveBigram = Counter()
    negativeBigram = Counter()
    for i in range(0, len(train_set)):
        for wordindex in range(0,len(train_set[i])-1):
            if train_labels[i] == 1: #positive
                positiveBigram[(train_set[i][wordindex], train_set[i][wordindex+1])] +=1
            if train_labels[i] == 0: #negative
                negativeBigram[(train_set[i][wordindex], train_set[i][wordindex+1])] +=1
    positiveBigram.subtract(positiveCounter)
    positiveBigram += Counter()
    negativeBigram.subtract(negativeCounter)
    negativeBigram += Counter()
    num_bigrams_positive = sum(positiveBigram.values())
    num_uniquebigrams_positive = len(list(positiveBigram))
    num_bigrams_negative = sum(negativeBigram.values())
    num_uniquebigrams_negative = len(list(negativeBigram))
    dev_labels = []

    for document in dev_set:
        positiveLikelihood = 0
        negativeLikelihood = 0
        positiveBigramLikelihood = 0
        negativeBigramLikelihood = 0
        for word in document:
            #gotcha: it is PLUS when taking the log
            positiveLikelihood += math.log((positiveCounter[word]+unigram_smoothing_parameter) \
            /(num_total_words_positive+unigram_smoothing_parameter*num_uniquewords_positive))
            negativeLikelihood += math.log((negativeCounter[word]+unigram_smoothing_parameter) \
            /(num_total_words_negative+unigram_smoothing_parameter*num_uniquewords_negative))
        positivePosterior = (1-bigram_lambda)*(positiveLikelihood+ math.log(pos_prior))
        negativePosterior = (1-bigram_lambda)*(negativeLikelihood+ math.log(1-pos_prior))
        for wordindex in range(0,len(document)-1):
            positiveBigramLikelihood += math.log((positiveBigram[(document[wordindex],document[wordindex+1])]+bigram_smoothing_parameter) \
            /(num_bigrams_positive+bigram_smoothing_parameter*num_uniquebigrams_positive))
            negativeBigramLikelihood += math.log((negativeBigram[(document[wordindex],document[wordindex+1])]+bigram_smoothing_parameter) \
            /(num_bigrams_negative+bigram_smoothing_parameter*num_uniquebigrams_negative))
        positivePosterior += bigram_lambda*(positiveBigramLikelihood+ math.log(pos_prior))
        negativePosterior += bigram_lambda*(negativeBigramLikelihood+ math.log(1-pos_prior))
        if (positivePosterior > negativePosterior):
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return dev_labels
