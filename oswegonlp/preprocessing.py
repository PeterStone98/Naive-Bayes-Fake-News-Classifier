from collections import Counter

import numpy as np
import pandas as pd

from oswegonlp.re_tokenizer import RegexTokenizer

# deliverable 1.1
def bag_of_words(text, retok):
    '''
    Count the number of word occurences for each document in the corpus

    :param retok: an instance of RegexTokenizer
    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    
    words = retok.tokenize(text)
    c = Counter(words)
   
    return c

# deliverable 1.2
def aggregate_word_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''
    
    counter = sum(bags_of_words, Counter())
        
    return counter
    
# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    uniqueWords = []
    
    for word in bow1:
        if word not in bow2:
            uniqueWords.append(word)
    
    return set(uniqueWords)

# deliverable 1.4
def prune_vocabulary(counts,x,threshold):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts
    
    :param counts: aggregated Counter for training data(Counter of all words with counts)
    :param x: list of Counters containing dev bow's(list of indiviual counters for each title)
    :return x_pruned: new list of Counters, with pruned vocabulary(list individual counters with words taken out of them)
    :return vocab: list of words in pruned vocabulary(Counter of all words with words taken out)
    :rtype: list of Counters, set
    '''
    lessThanThreshold = []
    allWords = []
        
    
    x_pruned = list()
    for counter in x:
        newCounter = Counter()
        newCounter = newCounter + counter
        for w in counter:
            if w not in allWords:
                allWords.append(w)
            if counts[w] < threshold:
                del newCounter[w]
                lessThanThreshold.append(w)
        x_pruned.append(newCounter)
    
    vocab = set([x for x in allWords if x not in lessThanThreshold])
  
    
    return x_pruned, vocab

# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
    
    #for each counter it has vocab
    #mark count of words in counter for row
    
    #doc1:03010000205
    #doc2:01900021300
    vocab = sorted(vocab)
    numpy_array = np.zeros((len(bags_of_words),len(vocab)))
    
    row=0
    for counter in bags_of_words:
        for word in counter:
            index = vocab.index(word)
            count = counter[word]
            numpy_array[row,index] = count
        row +=1
    return numpy_array
    
### Helper Code ###

def read_data(filename,label='RealOrFake',preprocessor=bag_of_words):
    retok = RegexTokenizer("[A-Za-z']+")
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string,retok) for string in df['Headline'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())

def isNotInt(s):
    try: 
        int(s)
        return False
    except ValueError:
        return True
    
def isGoodWord(s):
    noGood = ['the','that','in','at','it','is']
    if len(s) == 1:
        return False
    
    if s in noGood:
        return False
    
    return True
    