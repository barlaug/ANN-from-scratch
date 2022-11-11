#################################################################################################################################
## This file reads the weighs and biases, predicts the labels of the test data and evaluates the performance of the classifier ##
#################################################################################################################################
from utilities.file_utils import read_w_b
from utilities.train_utils import layers
from utilities.predict_evaluate_utils import predict, evaluate
from read_data import test_x, test_y

# Get weights and biases from file
weights, biases = read_w_b("weights_and_biases.txt", layers)

# Do prediction on test data and get prediction errors
y_pred, pred_errors = predict(test_x, test_y, weights, biases)

# Evaluate classifier performance
evaluate(y_pred, test_x, test_y, pred_errors)
