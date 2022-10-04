# This script does what it does

import helpers_matrix as hm
from read_data import train_x, train_y, test_x, test_y

num_inputs = 27
num_outputs = 1
layers = [num_inputs, num_inputs+num_outputs/2, num_outputs] # one hidden layer: mean of number of in and out nodes
