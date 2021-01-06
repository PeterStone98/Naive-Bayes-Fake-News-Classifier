from oswegonlp.constants import OFFSET
import numpy as np

# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# deliverable 2.1
def make_feature_vector(x,y):
    '''
    take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param x: counter of base features
    :param y: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    '''
    dictionary = {(y, OFFSET): 1}
    i = 1
    for w in x:
        dictionary[(y,w)]=i
        i = i + 1
        
    return dictionary

# deliverable 2.2
def predict(x,weights,labels):
    '''
    prediction function

    :param x: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    '''
    #go through x(pruned sinlge counter object)
    #for each word look up score in theta manual
    #add score to each label
    #compute top score
    maxCount = -1000 #arbitrary low number
    labelScores= {}
    topLabel = 'fake'
    
    for label in labels:
        if (label,OFFSET) in weights:
            labelScores[label] = weights[(label,OFFSET)]
        else:
            labelScores[label] = 0
       
    
    
    for label in labels:
        for word in x:
            weight = 0
            if (label,word) in weights:
                weight = weights[(label,word)]
            count = labelScores[label]
            newCount = count + weight
            labelScores[label] = newCount
            
    
    for label in labels:
        count = labelScores[label]
        if count > maxCount:
            maxCount = count
            topLabel = label
            
            
    if labelScores['fake'] == 0 and labelScores['real'] == 0:
        topLabel = 'fake'
    
    return topLabel, labelScores

def predict_all(x,weights,labels):
    '''
    Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    '''
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat