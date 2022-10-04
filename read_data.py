# This script reads the data from the data folder and creates training and testing data sets on the right format

path_dir = "data/"
filename_1 = "homology_reduced_subset_1.howlin"
filename_2 = "homology_reduced_subset_2.howlin"
filename_3 = "homology_reduced_subset_3.howlin"
filename_4 = "homology_reduced_subset_4.howlin"
filenames = [filename_1, filename_2]#, filename_3, filename_4] # include 3,4 later

# Use file 1,2,3 as training data, 4 as evaluation data

train_x = [] # inputs
train_y = [] # targets
test_x = []
test_y = []

for i, filename in enumerate(filenames):
    file = open(path_dir+filename, "r")
    f = file.readlines()

    if i == len(filenames)-1: # at end, use this file as test dataset
        for line in f:
            data = [float(el) for el in line.split()] # all elements of line.split() are strings, convert to floating point number
            test_x.append(data[:-1])
            test_y.append(int(data[-1])) # target value is always an integer

    else: # file content is to be part of the training dataset
        for line in f:
            # same approach as for the test dataset
            data = [float(el) for el in line.split()] 
            train_x.append(data[:-1])
            train_y.append(int(data[-1])) 

"""
# Testing. Remember to remove.
print("27?")
print(len(train_x[0]),len(train_x[100]), len(train_x[len(train_x)-1]))
print("sample:")
print(train_x[34])
print("equal?")
print(len(train_x))
print(len(train_y))

print("27?")
print(len(test_x[0]),len(test_x[100]), len(test_x[len(test_x)-1]))
print("sample:")
print(test_x[34])
print("equal?")
print(len(test_x))
print(len(test_y))
print(test_y)
"""

# SIZE OF TRAINING SET = 4045 SAMPLES
# SIZE OF TEST SET = 1344 SAMPLES

