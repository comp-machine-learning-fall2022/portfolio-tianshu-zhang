import pandas as pd
import numpy as np
from scipy.spatial import distance

# zero-center the data adapted from lab 5 code
def standardize(data):
    data_std = data.copy()

    # compute the mean and standard deviation of each column
    mean_vec = np.mean(data, axis=0)
    sd_vec = np.std(data, axis=0)

    # loop over the columns to standardize the data
    for i in range(data.shape[1]):
        data_std[:, i] = (data[:, i] - mean_vec[i] * np.ones(data.shape[0])) / sd_vec[i]
    return data_std

# perform k-means on the input data and return k centers and corresponding labels
def my_kmeans(array, cluster_num, rs):
    '''
    input: numpy array, cluster number k, random state
    return: the centers of the k clusters
    '''
    df = pd.DataFrame(array)

    # randomly assign initial centers
    centers = df.sample(cluster_num, random_state=rs).to_numpy()
    # set the number of max iterations
    max_it = 500

    for it in range(max_it):

        # label the datapoints to the nearest center
        dist = distance.cdist(array, centers)
        label = np.argmin(dist, axis=1)

        # initialize the new centers
        new_center = np.ones((cluster_num, array.shape[1]))

        # for each cluster, calculate the average date point as the new center
        for i in range(cluster_num):
            cluster = array[label == i]
            new_center[i] = np.mean(cluster, axis=0)

        # break if there is no change to the centers
        if (new_center == centers).all():
            break
        # update the centers
        else:
            centers = new_center

    return centers, label