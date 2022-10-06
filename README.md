# ANN from scratch
This project is part of course 22110 at Denmarks Technical University (DTU). The project aims at creating a well functioning artificial neural network (ANN) from scratch, using pure python. That is, no NumPy, Pandas, Keras, Tensorflow or any other supporting libraries. 

### The data:
The data is a part of a project at DTU HealthTech, which is about predicting whether certain variations of a Single-nucleotide polymorphism (SNP) will lead to a disease or not. The data folder consists of four files in which each line is a single datapoint. Three of the data sets are used for training and one is used for model evaluation.

The input for training is whitespace separated floating point numbers. There are 27 input variables and a single target value (1 = disease, 0 = healthy) as such:
```
0.075 0.3 0.075 0.225 -0.3 -0.075 0.15 -0.15 0 -0.3 -0.3 0.15 -0.3 -0.225 0 -0.15 -0.15 -0.3 -0.225 -0.3 -0.075 0.3 0.193746064573032135 1.011 0.7525 0.744312561819980218 0.612 0
-0.075 0.075 0.15 0.15 -0.375 -0.15 0.375 -0.15 -0.225 -0.375 -0.15 0.15 -0.3 -0.375 -0.3 -0.225 -0.225 -0.375 -0.3 -0.375 -0.225 0.15 0.173058043883721677 1.011 0.7525 0.744312561819980218 0.340 0
-0.3 -0.225 -0.15 -0.3 -0.45 -0.15 -0.225 -0.375 0.75 -0.375 -0.075 -0.225 -0.3 -0.3 -0.375 -0.3 -0.3 -0.375 -0.075 -0.375 -0.075 0.75 0.336279812283658667 1.011 0.7525 0.744312561819980218 0.100 1
-0.15 -0.3 0 -0.3 -0.225 0.075 -0.225 -0.375 0.6 -0.075 -0.15 -0.225 0 0.3 -0.225 -0.3 -0.3 -0.225 0.3 -0.3 -0.15 0.6 0.303728369615393505 1.011 0.7525 0.744312561819980218 0.212 1
```

### Modules:
* ***helpers_matrix.py***: Consists solely of utility functions for matrix and vector operations. 
* ***read_data.py***: Reads the data, transforms it into a the correct datatype and shape, and divides it into training and test datasets.
* ***train.py***: Initializes, creates and trains the neural net based on the training data. The model is implemented with a bias neuron on each layer - that way it learns better.
* ***predict_evaluate.py***: Predicts the test data by using the tuned model obtained from *train.py* and evaluates those predictions.

### Additional reading:
For improved understanding, see the following sources: 
* [Youtube series about deep learning basics](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* [Short high level article explaining the key concepts of an ANN](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/?fbclid=IwAR2vfPEcpnQ-Nl0ZXi-FqYZRHfFb9kzOZFGPktrXxuELaIDLA4NDPjs17RI)
* [Some calculus behind the backpropogation algorithm](https://towardsdatascience.com/neural-networks-backpropagation-by-dr-lihi-gur-arie-27be67d8fdce)
* [Explanation of stochastic gradient descent](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31)

### End note
The main focus of this project is to build a well functioning artificial neural network - tuning hyperparameters and optimizing the accuracy of the model has therefore only been done to a certain satisfactory level. The prediction errors reflects this fact.


**By:** Eirik Runde Barlaug and Sara Heiberg
