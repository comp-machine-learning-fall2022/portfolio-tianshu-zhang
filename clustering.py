import pandas as pd
import numpy as np

# zero-center the data adapted from lab 5 code
def standardize(data):
    data_std = data.copy()

    mean_vec = np.mean(data, axis=0)
    sd_vec = np.std(data, axis=0)

    for i in range(data.shape[1]):
        data_std[:, i] = (data[:, i] - mean_vec[i] * np.ones(data.shape[0])) / sd_vec[i]
    return data_std