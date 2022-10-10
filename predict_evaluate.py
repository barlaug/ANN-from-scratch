#TODO: check if output node should be changed somehow... see comment
#TODO: add plots of error_traj, acc_traj and so on, add some info to plot like alpha and such 

from train import weights, biases
from train import feed_forward
from read_data import test_x, test_y

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
        pred_errors: mean square error for each prediction of x_test
    """
    pred_errors = []
    predictions = []
    for i, sample_x in enumerate(x_test):
        sample_y = y_test[i]
        pred = feed_forward(sample_x, weights, biases)
        if pred > 0.5: # Naive? Change to some sort of softmax later on? -- Find out later maybe we dont have to put it in a class even
            predictions.append(1)
        else:
            predictions.append(0)
        error_i = 0.5*((pred - sample_y)**2)
        pred_errors.append(error_i)
    return predictions, pred_errors

preds, pred_errors = predict(test_x, test_y, weights, biases)
tot_errors = sum(pred_errors)/len(test_x)

# Count correct and false ones
correct = 0
for i in range(len(preds)):
    if preds[i] == test_y[i]:
        correct += 1
false = len(test_x) - correct

print(f"The average error on the test data is: {tot_errors}")
print(f"Number of correct classified input data points is: {correct}.\t That is {(correct/len(test_x))*100}% accurate.")
print(f"Number of incorrect classified input data points is: {false}.\t That is {(false/len(test_x))*100}% inaccurate.\n")
