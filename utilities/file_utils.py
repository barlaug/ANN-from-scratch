##########################################################################################################################################
## This file consists of two utility functions for writing and reading the weights and biases to/from the file "weights_and_biases.txt" ##
##########################################################################################################################################

def write_w_b(filename, weights, biases):
    """Writes thw weights and biases of the net to a file 'filename'"""
    f = open(filename, "w")
    w_mat, w_vec = weights[0], weights[1]
    b_vec, b_ = biases[0], biases[1]

    # write matrix to file, row wise
    for w_vec_i in w_mat:
        str_i = " " # init empty string
        w_vec_i_ = [str(w) for w in w_vec_i] # turn all elements of weight vector into string
        w_vec_i_str = str_i.join(w_vec_i_) # make above string list into a single string, separated by a whitespace
        f.write(w_vec_i_str + "\n") # write the string to file

    # write the rest row wise
    # weight vector:
    str_ = " "
    w_vec_ = [str(w) for w in w_vec]
    w_vec_str = str_.join(w_vec_) 
    f.write(w_vec_str + "\n")

    # bias vector:
    str_ = " "
    b_vec_ = [str(b) for b in b_vec]
    b_vec_str = str_.join(b_vec_) 
    f.write(b_vec_str + "\n")

    # single bias:
    b_str = str(b_)
    f.write(b_str + "\n")
    

def read_w_b(filename, layers):
    """Reads the weights and biases from a .txt-file and returns the weight- and bias matrices"""
    f = open(filename, "r")
    lines = f.readlines()

    w1 = []
    w2 = []
    b1 = []
    b2 = 0

    for i,line in enumerate(lines):
        if i < layers[1]: # weight matrix from 1st to 2nd layer
            w_i = line.strip().split(" ")
            w_i = [float(w) for w in w_i]
            w1.append(w_i)
        elif i == (layers[1]): # weight vector between 2nd and output-layer
            w2 = line.strip().split(" ")
            w2 = [float(w) for w in w2]
        elif i == (layers[1] + 1): # bias vector for 2nd layer
            b1 = line.strip().split(" ")
            b1 = [float(b) for b in b1]
        else: # we are at bottom of file
            b2 = line.strip()
            b2 = float(b2)
    
    w = [w1, w2]
    b = [b1, b2]

    return w, b