import pandas as pd
import numpy
from classification import *

# import data
corporate_pd = pd.read_csv("../corporate_rating.csv")

# convert the ratings to numerical values for analysis
ratings = corporate_pd['Rating'].unique()
values = [3,4,2,5,6,7,10,8,1,9]
corporate_pd['Rating'].replace(ratings,values,inplace=True)

# keep only the numerical factors
corporate_np = corporate_pd.drop(['Name','Symbol','Rating Agency Name','Date','Sector'],axis = 1).to_numpy()

train, test = split_train_test(corporate_np)

# test the shape of the test set
def test_test_split_shape():
    assert test.shape == (202, 26)

# test the type of the test set
def test_test_split_type():
    assert type(test) == numpy.ndarray

# # test the value of the mse calculation
# def test_mse_value():
#     mse = classification_mse(out_match, preds)
#     assert mse >=0 and mse <= 1
#
# # test the type of the mse calculation
# def test_mse_type():
#     mse = classification_mse(out_match, preds)
#     assert type(mse) == float
#
# # test the value of the CV error
# def test_CV_error_value():
#     error = kfold_CV(train, dt, 10)
#     assert error >= 0 and error <= 1
#
# # test the type of the CV error
# def test_CV_error_type():
#     error = kfold_CV(train, dt, 10)
#     assert type(error) == numpy.float64