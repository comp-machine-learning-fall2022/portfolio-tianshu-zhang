import pandas as pd
import numpy as np
import random
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# standardize data for dimension reduction
def standardize(data):
    data_std = data.copy()
    # mean and variance vectors
    mean_vec = np.mean(data, axis=0)
    sd_vec = np.std(data, axis=0)
    # standaridze the data by subtracting off the mean and divide by variance
    for i in range(data.shape[1]):
        data_std[:, i] = (data[:, i] - mean_vec[i] * np.ones(data.shape[0])) / sd_vec[i]

    return data_std

# split the data into 90% train 10% test
def split_train_test(data):
    np.random.seed(0)
    # sample 10% of the total number of indices
    index = np.random.choice(len(data), size = len(data) // 10, replace = False)
    # save 10% in the test set
    test = data[index,:]
    # save the rest in the training set
    train = np.delete(data, index, axis = 0)
    return train,test

# cross validation of classification prediction
def classification_mse(class_truth, pred_class):
    error = 0
    for i in range(len(class_truth)):
        if class_truth[i] != pred_class[i]:
            error = error + 1
    return error/len(class_truth)

# k fold cross-validation for specified model
def kfold_CV(data, model, k):
    # set train and test data
    n = math.floor(len(data) / k)

    # creat fold splits within the data
    splits = [i * n for i in range(1 + k)]
    # compute residual size with fold size n
    residual = len(data) % k
    if residual != 0:
        # add an extra element to each fold for the first residual number of folds
        splits = [splits[i] + min(i, residual) for i in range(1, k + 1)]
        # add 0 as the begin split index
        splits.insert(0, 0)

        # initialize list to store training and testing error
    train_errors = []
    test_errors = []
    for i in range(k):
        train = np.delete(data, range(splits[i], splits[i + 1]), axis=0)
        test = data[splits[i]:splits[i + 1], :]
        # Fit model to training dataset
        mod = model.fit(train[:, 1:], train[:, 0])

        # Compute the training error
        train_preds = mod.predict(train[:, 1:])
        train_error = classification_mse(train_preds, train[:, 0])
        train_errors.append(train_error)

        # Compute the testing error
        test_preds = mod.predict(test[:, 1:])
        test_error = classification_mse(test_preds, test[:, 0])
        test_errors.append(test_error)
    cross_val_error = np.mean(test_errors)
    return cross_val_error

# take the difference between two 2D numpy arrays
# https://stackoverflow.com/questions/66674537/python-numpy-get-difference-between-2-two-dimensional-array
def setdiff2d_list(arr1, arr2):
    delta = set(map(tuple, arr2))
    return np.array([x for x in arr1 if tuple(x) not in delta])

# sample 10 data points from each of the 10 credit ratings in the data as labeled data
def kNN_sample(data2D):
    data_df = pd.DataFrame(data2D)
    data_df.rename(columns={0: 'ratings'}, inplace=True)

    # sample 1 data point from each rating
    answers = data_df.groupby('ratings').sample(1).to_numpy()

    # split the data into class labels and 2D coordinates
    label_data = answers[:, 1:]
    pt_label = answers[0:, 0]

    # exclude the labeled data from the inital data to fit the model
    unlabeled = setdiff2d_list(data2D, answers)

    return label_data, pt_label, unlabeled


# kfold cross-validation for kNN with specified number of neighbors, 2D input data, and k folds
def kNN_kfold_CV(neighbor, data, k):
    # set train and test data
    n = math.floor(len(data) / k)

    # creat fold splits within the data
    splits = [i * n for i in range(1 + k)]
    # compute residual size with fold size n
    residual = len(data) % k
    if residual != 0:
        # add an extra element to each fold for the first residual number of folds
        splits = [splits[i] + min(i, residual) for i in range(1, k + 1)]
        # add 0 as the begin split index
        splits.insert(0, 0)

        # initialize list to store training and testing error
    train_errors = []
    test_errors = []
    for i in range(k):
        train = np.delete(data, range(splits[i], splits[i + 1]), axis=0)
        test = data[splits[i]:splits[i + 1], :]

        label_data, pt_label, unlabeled = kNN_sample(train)
        # Fit model to training dataset
        model = KNeighborsClassifier(n_neighbors=neighbor)
        mod = model.fit(label_data, pt_label)

        # Compute the training error
        train_preds = mod.predict(unlabeled[:, 1:])
        train_error = classification_mse(train_preds, unlabeled[:, 0])
        train_errors.append(train_error)

        # Compute the testing error
        test_preds = mod.predict(test[:, 1:])
        test_error = classification_mse(test_preds, test[:, 0])
        test_errors.append(test_error)
    cross_val_error = np.mean(test_errors)
    return cross_val_error