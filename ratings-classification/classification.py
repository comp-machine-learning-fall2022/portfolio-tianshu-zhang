import pandas as pd
import numpy as np

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


def kNN_sample(data2D):
    data_df = pd.DataFrame(data2D)
    data_df.rename(columns={0: 'ratings'}, inplace=True)
    # sample 1 data point from each rating
    answers = data_df.groupby('ratings').sample(1).to_numpy()

    # split the data into class labels and 2D coordinates
    label_data = answers[:, 1:]
    pt_label = answers[0:, 0]

    # exclude the labeled data from the inital data to fit the model
    unlabeled = np.setdiff1d(data2D, answers).reshape((1817, 2))

    return label_data, pt_label, unlabeled