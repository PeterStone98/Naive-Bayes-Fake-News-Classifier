import functools
import itertools

from oswegonlp.constants import OFFSET
from oswegonlp import classifier_base, evaluation, preprocessing 


import numpy as np
from collections import defaultdict, Counter

# deliverable 3.1
def corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    #go through each word and if its document is labeled fake add fake count
    dictionary = defaultdict(float)
    length = len(x)
    
    
    n=0
    for i in range(0,length):
        if y[i] == label:
            for word in x[i]:
                if word in dictionary:
                    count = dictionary[word]
                    newCount = count + x[i][word]
                    dictionary[word]= newCount
                if word not in dictionary:
                    dictionary[word] = x[i][word]
                    
    return dictionary

# deliverable 3.2
def estimate_pxy(x,y,label,alpha,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance(list of counters)
    :param y: list of labels, one per instance(list of labels)
    :param label: desired label
    :param alpha: additive smoothing amount(just a number)
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

       ( count of word in label )
  log  ( ---------------------- )
       (  total words in label  )

    '''
    V = len(vocab)
    dictionary = defaultdict(float)
    totalTokens = 0
    labelCounts = corpus_counts(x,y,label)
    
    for i in range(0,len(x)):
        if y[i] == label:
             for word in x[i]:
                totalTokens = totalTokens + x[i][word]
           
    
    for word in vocab:
        lcount = labelCounts[word]
        dictionary[word] = np.log((alpha + lcount)/(alpha*V + totalTokens))
    
    
    return dictionary

# deliverable 3.3
def estimate_nb(x,y,alpha):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts(bag of words)
    :param y: list of labels()
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    
    labels = set(y)
    labelCounts = Counter(y)#get counts of each label
    
    doc_counts = defaultdict(float)
    
    #get vocab
    Vocab = preprocessing.aggregate_word_counts(x)
    
    for label in labels:
        #get offsets for labels
        offset = labelCounts[label]/len(y)
        doc_counts[(label,OFFSET)] = np.log(offset)
       
        #compute prbability for words with label
        dictionary = estimate_pxy(x,y,label,alpha,Vocab.keys())
        
        #add all probabilities
        for d in dictionary:
            doc_counts[(label,d)] = dictionary[d]
    
    
    return doc_counts

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,alphas):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training (bag of words)
    :param y_tr: training labels(predicted labels)
    :param x_dv: dev instances(bag of words)
    :param y_dv: dev labels(actual labels)
    :param alphas: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''
    bestValue = 0.0
    bestScore = 0.0
    scores = defaultdict(float)
    labels = set(y_tr)
    
    for alpha in alphas:
        #compute naive bayes
        theta_nb = estimate_nb(x_tr,y_tr,alpha)
        
        #compute labels using predict all
        predicted_labels = classifier_base.predict_all(x_dv,theta_nb,labels)
        
        #compute accuracy
        score = evaluation.acc(predicted_labels,y_dv)
        scores[alpha]=score
        
        #check score
        if score > bestScore:
            bestScore = score
            bestValue = alpha
        
    
    return bestValue, scores

