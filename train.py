###################################################################################################################################
## This file initializes and trains the neural net, it then calculates the training error and writes weights and biases to file. ##
###################################################################################################################################
from utilities.file_utils import write_w_b
from utilities.train_utils import *
from read_data import train_x, train_y

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


# Write weights and biases to file "weights_and_biases.txt"
write_w_b("weights_and_biases.txt", weights, biases)

print(weights, biases)


# PLOTTING: 
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