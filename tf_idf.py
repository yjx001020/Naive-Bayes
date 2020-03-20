# tf_docs_have_this_word_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter
import time



def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-docs_have_this_word value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """

    best_tf_idf_words = []
    docs_have_this_word = Counter()
    for review in train_set:
        c = Counter()
        for word in review:
                c[word] += 1
        for word in c:
                if c[word] > 0:
                    docs_have_this_word[word] += 1

    for review in dev_set:
        words = Counter()
        for word in review:
            words[word] += 1
        num_total = sum(words.values())
        for word in words:
            tf_idf = (words[word] / num_total) * math.log(len(train_set) / (1 + docs_have_this_word[word]))
            words[word] = tf_idf
        best_tf_idf_words.append(words.most_common(1)[0][0])
    for i in range(0,20):
        print(best_tf_idf_words[i])
    return best_tf_idf_words
