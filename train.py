import helpers_matrix as hm
from read_data import train_x, train_y
import random
import math 

def init_weights_and_bias(layer1, layer2):
    """Generate weight matrix and bias vector between two layers
    params: 
        layer1, layer2: size of two layers to make weights between
    returns:
        w: weight matrix of size (len(layer2), len(layer1))
        b: bias weight vector of size (1, len(layer 2))
    """
    w = []
    b = []
    for _ in range(layer2):
        w_i = []
        b_i = random.gauss(0,1)
        for _ in range(layer1):
            w_j = random.gauss(0,1) # take one sample from the standard normal dist.
            w_i.append(w_j)
        w.append(w_i)
        b.append(b_i)

    if len(w) == 1: # is the case if weight is for a single output node
        return w[0], b[0]
    else: 
        return w, b

# Activation function: Sigmoid.
def sigmoid(x):
    """Returns the sigmoid function evaluated at x"""
    return 1 / (1 + math.exp(-x))

def feed_forward(x, weights, biases):
    """Feeds forward the observation x through the network, given the weights and biases on the edges in the net
    params:
        x: input observation, vector of length n
        weights: list of weight matrices between layer i and layer i+1
    returns:
        a2: value at the end node (output of the second layer), float
    """
    # Since our neural net is quite basic, we can do this without almost any loops and logic, which improves readability

    # Unpack weights ans biases
    w1 = weights[0]
    w2 = weights[1]
    b1 = biases[0]
    b2 = biases[1]
    
    # Hidden layer outputs = a1
    z1 = [] 
    for i, w in enumerate(w1):
        z1.append(hm.dot(x,w) + b1[i])
    a1 = [sigmoid(el) for el in z1] 
    
    # Final layer output = a2
    z2 = hm.dot(a1,w2) + b2
    a2 = sigmoid(z2)
    return a2


def backpropagate(x, y, weights, biases, a):
    """Updates weight matrices based on the error between the feedforward of x and the target y, using backpropagation
    params:
        x: input vector
        y: target
        weights: list of weight matrices between layer i and layer i+1
        biases: list of bias vectors between layer i and layer i+1
        a: alpha, learning rate
    returns:
        weights_new: list of updated weight matrices
        biases_new: list of updated bias vectors
    """
    # Unpack weights and biases
    w1 = weights[0]
    w2 = weights[1]
    b1 = biases[0]
    b2 = biases[1]

    # Hidden layer
    z1 = []
    for i, w in enumerate(w1):
        z1.append(hm.dot(x,w) + b1[i])
    a1 = [sigmoid(el) for el in z1] 

    # Output layer
    z2 = hm.dot(a1,w2) + b2
    a2 = sigmoid(z2)

    # Error and derivatives. See: https://towardsdatascience.com/neural-networks-backpropagation-by-dr-lihi-gur-arie-27be67d8fdce
    # We compute four gradients to update the weight and bias matrices/vectors. These are named as follows in the code:
    # 1: dJ/dw2 = dj_dw2, 2: dJ/db2 = dj_db2
    # 3: dJ/dw1 = dj_dw1, 4: dJ/db1 = dj_db1
    # To arrive at these gradients, the chain rule is applied. All the in-between calculations follow the same naming scheme.
    # We will also keep track of the variable dimention as a comments, to make the calculus more clear when reading the code.
    
    # Error function = J = 0.5*((a2-y)**2) 
    
    # 1. dJ/dw2
    dj_da2  = (a2-y) # scalar
    da2_dz2 = a2*(1-a2) # scalar
    dz2_dw2 = a1 # vector
    dj_dw2 = hm.scalar_mult(dz2_dw2, (dj_da2*da2_dz2)) # vector

    # 2. dJ/db2
    dz2_db2 = 1
    dj_db2 = dj_da2*da2_dz2*dz2_db2 # scalar

    # 3. dJ/dw1
    dz2_da1 = w2 # vector
    # The second argument in the dot function below is simply (1-a1) where "1" is a vector of ones with the same length as a1
    da1_dz1 = hm.dot(a1, [(1-el) for el in a1]) # scalar
    dz1_dw1 = x # vector

    # Now comes a fairly long expression, we therefore divide it into subcalcluations, with corresponding subscripts in the variable name
    dj_dw1_1 = hm.scalar_mult(dz2_da1, dj_da2*da2_dz2) # vector
    dj_dw1_2 = hm.scalar_mult(dj_dw1_1, da1_dz1) # vector

    # For dj/dw1 we want a (len(layer2)xlen(layer1))-matrix as this is the shape of w1. 
    # Since dj_dw1_2 is a (1xlen(layer2))-vector and dz1_dw1 is a (1xlen(layer1))-vector, we will perform the calculation 
    # dot(transpose(dj_dw2_2), dz1_dw1) -- The following loop does so:
    dj_dw1 = [] 
    for row_i in dj_dw1_2:
        matrix_row_i = []
        for col_j in dz1_dw1:
            matrix_row_i.append(row_i*col_j)
        dj_dw1.append(matrix_row_i)

    # dj_dw1 is now a (len(layer2)xlen(layer1))-matrix as it should

    # 4. dJ/db1
    dj_db1 = dj_dw1_2 # vector

    # Weight update (a is learning rate)
    w1_new = hm.diff(w1, hm.scalar_mult(dj_dw1, a))
    b1_new = hm.diff(b1, hm.scalar_mult(dj_db1, a))
    w2_new = hm.diff(w2, hm.scalar_mult(dj_dw2, a))
    b2_new = b2 - (a*dj_db2)

    weights_new = [w1_new, w2_new]
    biases_new  = [b1_new, b2_new]

    return weights_new, biases_new


 #insert function to shuffle x?

def train(x_train, y_train, weights, biases, a, num_epochs=100, tol_err=0.05):
    """Trains the neural net on all xi in x_train against all targets in y_train
    params:
        x_train: all training observations
        y_train: all training targets (binary)
        weights: list of weight matrices between layer i and layer i+1
        biases: list of bias vectors between layer i and layer i+1
        a: learning rate
        num_epochs: number of times to iterate through all the traing data
        tol: tolerance for change in error from one iteration to another
    returns:
        error_trajectory: average error for all iterations in num_iterations
        weight: tuned weights
        biases: tuned biases
    """
    error_trajectory = [] # keep track of the loss at output
    i = 0
    while (i < num_epochs):
        errors_i = []
        for k, xk in enumerate(x_train):
            yk = y_train[k]
            # First, feed through and evaluate:
            err = 0.5*((feed_forward(xk, weights, biases) - yk)**2)
            errors_i.append(err)
            # Then, backpropagate and update weights/biases:
            weights, biases = backpropagate(xk, yk, weights, biases, a)
        
        mean_error_i = sum(errors_i)/len(x_train)
        error_trajectory.append(mean_error_i)
        print(f"Epoch: {i}, Avg. error: {mean_error_i}")
        if mean_error_i < tol_err:
            print(f"Finished. Error below tolerance ({tol_err}).")
            break
        
        # Shuffle x_train and y_train equally
        xy_tuples = list(zip(x_train, y_train))
        random.shuffle(xy_tuples)
        x_train, y_train = zip(*xy_tuples)

        i += 1
    
    return error_trajectory, weights, biases


# MODEL:
import matplotlib.pyplot as plt

# Define number of layers and number of nodes per layer
num_inputs = len(train_x[0])
num_outputs = 1
layers = [num_inputs, int((num_inputs+num_outputs)/2), num_outputs] # one hidden layer: size = mean of number of in and out nodes

# Initialize weight and biases
w1, b1  = init_weights_and_bias(layers[0], layers[1])
w2, b2  = init_weights_and_bias(layers[1], layers[2])
weights = [w1, w2]
biases  = [b1, b2]

# Create and train model
alpha = 0.005
n_epochs = 50
error_traj, weights, biases = train(train_x, train_y, weights, biases, a=alpha, num_epochs=n_epochs)
accuracy_traj = [(1-el) for el in error_traj]

# PLOTS: 
"""
import matplotlib.pyplot as plt
n_epochs = 50
error_traj_1, _, _ = train(train_x, train_y, weights, biases, a=0.1, num_epochs=n_epochs)
error_traj_2, _, _ = train(train_x, train_y, weights, biases, a=0.01, num_epochs=n_epochs)
error_traj_3, _, _ = train(train_x, train_y, weights, biases, a=0.005, num_epochs=n_epochs)
error_traj_4, _, _ = train(train_x, train_y, weights, biases, a=0.001, num_epochs=n_epochs)

plt.plot(error_traj_1)
plt.plot(error_traj_2)
plt.plot(error_traj_3)
plt.plot(error_traj_4)
plt.xlabel("epoch")
plt.ylabel("Avg. error")
plt.legend(["alpha = 0.1", "alpha = 0.01", "alpha = 0.005", "alpha = 0.001"])
plt.title("Error trajectories for different alphas. n_epoch = 50.")
plt.savefig("alpha_comparison_6", dpi=400)
plt.show()
"""