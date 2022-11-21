#######################################################################
## This file contains utilities for the predictions on the test data ##
#######################################################################
from utilities.train_utils import feed_forward

def predict(x_test, y_test, weights, biases):
    """Predicts the labels y_test frm the data x_test with weight and biases
       from trained model.
    params:
        x_test: test data
        y_test: test labels
        weights: weights obtained in training
        biases: biases obtained in training
    returns:
        predictions: predictions for all instances in x_test
        pred_mse: mean square error for each prediction of x_test
    """
    pred_mse = []
    predictions = []
    for i, sample_x in enumerate(x_test):
        sample_y = y_test[i]
        pred = feed_forward(sample_x, weights, biases)
        if pred > 0.5:
            predicted_class = 1
        else:
            predicted_class = 0
        predictions.append(predicted_class)
        error_i = 0.5*((pred - sample_y)**2)
        pred_mse.append(error_i)
    return predictions, pred_mse

def evaluate(y_pred, x_test, y_test, pred_errors):
    """Evalueates the performance of the classifier based on the predicted and true labels. Prints results."""

    tot_errors = sum(pred_errors)/len(x_test)
    # Count correct and false ones
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            correct += 1
    false = len(x_test) - correct
    print(f"The per-epoch-average mean square error on the test data is: {round(tot_errors, 6)}")
    print(f"Number of correct classified input data points is: {correct}. That is {round((correct/len(x_test))*100, 4)}% accurate.")
    print(f"Number of incorrect classified input data points is: {false}. That is {round((false/len(x_test))*100, 4)}% inaccurate.")