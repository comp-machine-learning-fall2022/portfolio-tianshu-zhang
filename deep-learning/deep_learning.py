import pandas as pd
import numpy as np

# sample 10% of the total data in the test set
def split_train_test(data):
    np.random.seed(0)
    # sample 10% of the total number of indices
    index = np.random.choice(len(data), size = len(data) // 10, replace = False)
    # save 10% in the test set
    test = data[index,:]
    # save the rest in the training set
    train = np.delete(data, index, axis = 0)
    return train,test

# split the data into ratings and features
def split_rating_features(data):
    ratings = data[:,0]
    features = data[:,1:]
    return ratings, features

# calculate classification mse
def classification_mse(class_truth, pred_class):
    error = 0
    for i in range(len(class_truth)):
        if class_truth[i] != pred_class[i]:
            error = error + 1
    return error/len(class_truth)

