import pandas as pd
import numpy
from classification import *

# import data
corporate_pd = pd.read_csv("corporate_rating.csv")

# convert the ratings to numerical values for analysis
ratings = corporate_pd['Rating'].unique()
values = [3,4,2,5,6,7,10,8,1,9]
corporate_pd['Rating'].replace(ratings,values,inplace=True)

# keep only the numerical factors
corporate_np = corporate_pd.drop(['Name','Symbol','Rating Agency Name','Date','Sector'],axis = 1).to_numpy()

# split into train and test sets
train, test = split_train_test(corporate_np)

# standardize the training data
train_input = standardize(train[:,1:])
ratings = np.transpose(np.array([train[:,0]]))

# PCA dimension reduction
pca = PCA(n_components=2)
train_input_2D = pca.fit_transform(train_input)
# add back the ratings
train_2D = np.hstack((ratings, train_input_2D))

# split the data into info and label
info = train[:,1:]
label = train[:,0]

# random forest
forest = RandomForestClassifier(n_estimators=3, max_features = 8, max_depth=5, random_state=0)
forest.fit(info, label)
test_preds = forest.predict(test[:,1:])

label_data, pt_label, unlabeled = kNN_sample(train_2D)

# kNN with 3 neighbors
kNN_alg1 = KNeighborsClassifier(n_neighbors=3)
kNN_alg1.fit(label_data,pt_label)

# test standize function
def test_standardize_type():
    assert isinstance(standardize(train), np.ndarray)

def test_standardize_shape():
    assert standardize(train).shape == train.shape

# test the shape of the test set
def test_test_split_shape():
    assert test.shape == (202, 26)

# test the type of the test set
def test_test_split_type():
    assert type(test) == numpy.ndarray

# test the value of the mse calculation
def test_mse_value():
    mse = classification_mse(test[:,0], test_preds)
    assert mse >=0 and mse <= 1

# test the type of the mse calculation
def test_mse_type():
    mse = classification_mse(test[:,0], test_preds)
    assert type(mse) == float

# test the value of the CV error
def test_CV_error_value():
    error = kfold_CV(train, forest, 10)
    assert error >= 0 and error <= 1

# test the type of the CV error
def test_CV_error_type():
    error = kfold_CV(train, forest, 10)
    assert type(error) == numpy.float64

# test the set difference shape
def test_setdiff2d_list():
    assert setdiff2d_list(train_2D, unlabeled).shape == (10,3)

# test the value of the CV error for kNN
def test_kNN_kfold_CV_error_value():
    error = kNN_kfold_CV(3, train_2D, 10)
    assert error >= 0 and error <= 1

# test the type of the CV error for kNN
def test_kNN_kfold_CV_error_type():
    error = kNN_kfold_CV(3, train_2D, 10)
    assert type(error) == numpy.float64

# test the shape of kNN_sample output
def test_kNN_sample_shape():
    res = kNN_sample(train_2D)
    assert len(res) == 3

# test shape of label data and pt_label
def test_kNN_sample_label_data_shape():
    label_data, pt_label, unlabeled = kNN_sample(train_2D)
    assert label_data.shape == (10,2) and len(pt_label) == 10