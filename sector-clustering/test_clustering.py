import pandas as pd
import numpy as np
from clustering import *
from sklearn.decomposition import PCA

# import data
corporate_pd = pd.read_csv("corporate_rating.csv")
corporate_np = corporate_pd.drop(['Rating','Name','Symbol','Rating Agency Name','Date','Sector'],axis = 1).to_numpy()
sectors = corporate_pd[['Sector']].to_numpy()

# standardize data
corporate_std = standardize(corporate_np)

# visualize the data in 2D
pca = PCA(n_components=2)
corporate2D = pca.fit_transform(corporate_std)

sectorlist = list(corporate_pd.Sector.unique())
sector_num = sectors.copy()
for i in range(len(sectors)):
    sector_num[i] = sectorlist.index(sectors[i])

# eliminate outliers for visualization
# keep data points with x < 0.5
include = corporate2D[:,0]<0.5
corporate2Dnew = corporate2D[include]
sector_num1 = sector_num[include]

# keep data points with x > -0.5
include1 = corporate2Dnew[:,0]>-0.5
corporate2Dnew1 = corporate2Dnew[include1]
sector_num2 = sector_num1[include1]

# keep data points with y < 1
include2 = corporate2Dnew1[:,1]<1
corporate2Dnew2 = corporate2Dnew1[include2]
sector_num3 = sector_num2[include2]

# test standize function
def test_standardize_type():
    assert isinstance(standardize(corporate_np), np.ndarray)

def test_standardize_shape():
    assert standardize(corporate_np).shape == corporate_np.shape

# test my_kmeans function (adapted from hw 2)
def test_my_kmeans_type():
    assert isinstance(my_kmeans(corporate2Dnew2, 3, 2019), tuple)

def test_my_kmeans_shape():
    expected = 2
    assert len(my_kmeans(corporate2Dnew2, 3, 2019)) == expected

def test_my_kmeans_center_num():
    expected = (3, 2)
    centers_shape = my_kmeans(corporate2Dnew2, 3, 2019)[0].shape
    assert centers_shape == expected

def test_my_kmeans_labels():
    expected = 2
    label_max = np.max(my_kmeans(corporate2Dnew2, 3, 2019)[1])
    assert label_max == expected

# test looping k-means
def test_looping_kmeans_type():
	assert isinstance(looping_kmeans(corporate2Dnew2,list(range(1,15))), list)

def test_looping_kmeans_size():
	expected = 14
	assert len(looping_kmeans(corporate2Dnew2,list(range(1,15)))) == expected

def test_looping_kmeans_goodness():
	out = looping_kmeans(corporate2Dnew2,list(range(1,15)))
	assert (out[1:] <= out[:-1])