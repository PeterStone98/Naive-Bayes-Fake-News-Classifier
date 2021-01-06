from oswegonlp.constants import OFFSET
from collections import Counter
from torch.autograd import Variable
import numpy as np
import torch

# deliverable 6.1
def get_top_features_for_label_numpy(weights,label,k=5):
    '''
    Return the k features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    '''
    top_labels = [None] * k
    count =0
    
    sorted_list = sorted(weights.items(), key=lambda x:x[1], reverse=True)
    for weight in sorted_list:
        l = str(weight)[3:]
        if label == l[:4]:
            top_labels[count] = weight
            count += 1
            if count == k:
                break
            
    
    
    return top_labels


# deliverable 6.2
def get_top_features_for_label_torch(model,vocab,label_set,label,k=5):
    '''
    Return the k words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted(vocab)
    :param label_set: set of ordered labels(real,fake)
    :param label: the label you are interested in(label)
    :returns: list of words
    :rtype: list
    '''
    
    top_weights = [0.0] * k
    top_labels = [''] * k
    index = label_set.index(label)
    vocab = sorted(vocab)
    weights = list(model.parameters())[0].data.numpy()#numpy array for each word it has real weight-fake weight
    print(weights)
    
    
    i=0
    for word in vocab:
        weight = weights[index][i]
        if (weight > top_weights).any:
            x=0
            for num in top_weights:#check if weight is greater than top weights
                if weight > num:
                    top_weights[x] = weight#change out weight for new weight
                    top_weights = sorted(top_weights)#sort weights
                    top_labels[top_weights.index(weight)] = vocab[i]#allocate word to index slot
                    break
                x += 1      
        i += 1 
        
    print(top_weights)
    
    
    return top_labels
