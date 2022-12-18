import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
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


# perform k-means by looping aver a list of k values (adapted from homework 2)
def looping_kmeans(array, klist):
    # initialize output list
    goodness = [0] * len(klist)

    # implement sklearn k-means for each k value in the list
    for k in klist:
        km_alg = KMeans(n_clusters=k, init="random", random_state=2, max_iter=200)
        fit = km_alg.fit(array)
        centers = fit.cluster_centers_
        labels = fit.labels_

        # initialize total distance -- goodness of fit
        cluster_total = 0
        for i in range(k):
            clusteri = array[labels == i]
            # compute the distance of the point to cluster center
            cluster_spread = distance.cdist(clusteri, centers[[i]], 'euclidean')
            # take the sum of all the distances within the cluster
            cluster_sum = np.sum(cluster_spread)
            # add the total distance to the total cluster distances
            cluster_total += cluster_sum
        # store the sum of the distance as the goodness for k clusters
        goodness[i] = cluster_total

    return goodness