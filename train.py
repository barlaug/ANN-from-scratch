# This script does what it does

import helpers_matrix as hm
from read_data import train_x, train_y, test_x, test_y
import random
import math 

num_inputs = 27
num_outputs = 1
layers = [num_inputs, int((num_inputs+num_outputs)/2), num_outputs] # one hidden layer: mean of number of in and out nodes

def init_weights(layer1, layer2):
    """Generate weight matrix between layer 1 and layer 2, each row is input weight for one node in layer 2
    params: 
        layer1, layer2: size of two layers to make weights between
    returns:
        w: weight matrix of size (len(layer2), len(layer1))
    """
    w = []
    for i in range(layer2):
        w_i = []
        for j in range(layer1):
            w_j = random.gauss(0,1) # take one sample from the standard normal dist.
            w_i.append(w_j)
        w.append(w_i)

    if len(w) == 1: # is the case if weight is for a single output node
        return w[0]
    else: 
        return w

# Activation function: Sigmoid.
def sigmoid(x):
    """Returns the sigmoid function evaluated at x"""
    return 1 / (1 + math.exp(-x))

def feed_forward(x, weights):
    """Feeds forward the observation x through the network, given the weights on the edges between nodes
    params:
        x: input observation, vector of length n
        weights: list of weight matrices between layer i and layer i+1
    returns:
        out: output at the end node, float
    """
    # Since our neural net is quite basic, we can do this without almost any loops and logic, this improves readability
    # Unpack weight matrices
    w1 = weights[0]
    w2 = weights[1]
    
    # Hidden layer outputs
    out1 = []
    for w in w1:
        print(len(w))
        print(sigmoid(hm.dot(x,w)))
        out1.append(sigmoid(hm.dot(x,w)))
    
    # Output layer output
    out = sigmoid(hm.dot(out1, w2))
    return out

x = train_x[1]
hm.printlist(x)
w1 = init_weights(layers[0], layers[1])
w2 = init_weights(layers[1], layers[2])
print(w2)
weights = [w1, w2]
out = feed_forward(x, weights)
print(f"ouuuuut: {out}")




        
#hm.printlist(init_weights(layers[0], layers[1]))
