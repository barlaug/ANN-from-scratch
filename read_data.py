path_dir = "data/"
filename_1 = "homology_reduced_subset_1.howlin"
filename_2 = "homology_reduced_subset_2.howlin"
filename_3 = "homology_reduced_subset_3.howlin"
filename_4 = "homology_reduced_subset_4.howlin"
filenames = [filename_1, filename_2, filename_3, filename_4] 

# Use file 1,2,3 as training data, 4 as evaluation data

train_x = [] # training inputs
train_y = [] # training targets
test_x  = [] # testing inputs
test_y  = [] # testing targets

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


# WITH 1,2,3 AS TRAIN AND 4 AS TEST:

# SIZE OF TRAINING SET = 4045 SAMPLES
# SIZE OF TEST SET = 1344 SAMPLES