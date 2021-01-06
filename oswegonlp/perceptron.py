from collections import defaultdict
from oswegonlp.constants import OFFSET
from oswegonlp.classifier_base import predict,make_feature_vector
import numpy as np

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights(counter)
    :param y: label, a string(correct label)
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''
    dictionary = defaultdict(float)
    predicted_label,_ = predict(x,weights,labels)
    feature_vec = make_feature_vector(x,y)
   

    if predicted_label != y:
        for label in labels:
            update = -1
            if label == y:
                update = 1
            dictionary[(label,OFFSET)] = update
            for word in x:
                dictionary[(label,word)] = update
            
    

    return dictionary

# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights(small bag of words)
    :param y: label, a string(actual labels)
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''
    '''         if ^y != y(i) then
                    (theta)  = (theta-1) + f(x(i); y(i)) - f(x(i); ^y)
                else
                    (theta)  = (theta-1)'''
    
    #predict the label using
    #update labels recordingly
    
    #TRIED TO MAKE AN AVERAGE PERCEPTRON
    
    labels = set(y)
    weights = defaultdict(float)
    m = defaultdict(float)
    weight_history = []
    t=0
    for it in range(N_its):
        for x_i,y_i in zip(x,y):
            t+=1
            y_pred,_ = predict(x_i,weights,set(y))
            if y_pred != y_i:
                new_weights = perceptron_update(x_i,y_i,weights,labels)
                for key in new_weights:
                    weights[key] += new_weights[key] 
            for key in weights:
                m[key] += weights[key]
        weight_history.append(weights.copy())
        
    for k in m.keys():
        m[k] = m[k]/t
    
    
    return m, weight_history

def average_pereptron(x,y):
    return 1

