# ANN from scratch
This project is part of course 22110 at Denmarks Technical University (DTU). The project aims at creating a well functioning artificial neural network (ANN) from scratch, using pure python. That is, no NumPy, Pandas, Keras, Tensorflow or any other supporting libraries. 

### The data:
The data is a part of a project at DTU HealthTech, which is about prediction of whether certain variations of a Single-nucleotide polymorphism (SNP) will lead to a disease or not. The data folder consists of 4 .howlin-files in which each line is a datapoint. Three of the data sets are used for training and one is used for model evaluation.

## Input-Output:
The input for training is whitespace separated floating point numbers. There are 27 input variables and a single target value (1=disease, 0=health).

## Modules:
* ***helpers_matrix.py***: Consists solely of utility functions for matrix and vector operations. 
* ***read_data.py***: Reads the data, transforms it into a the correct datatype and shape, and divides it into training and test datasets.
* ***train.py***: Initializes, creates and trains the neural net based on the training data.
* ***predict_evaluate.p***: Predicts the test data by using the tuned model obtained from *train.py* and evaluates those predictions.

## NB
The main focus of this project is to build a well functioning artificial neural network - tuning hyperparameters and optimizing the accuracy of the model has therefore only been done to a certain satisfactory level. The prediction errors reflects this fact.


By Eirik Runde Barlaug and Sara Heiberg
