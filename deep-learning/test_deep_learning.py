from sklearn.neural_network import MLPClassifier
from deep_learning import *
import numpy

# import data
corporate_pd = pd.read_csv("../corporate_rating.csv")
corporate_pd = corporate_pd.drop(['Name','Date','Rating Agency Name','Sector'],axis = 1)
corporate_filtered = corporate_pd.groupby('Symbol').filter(lambda x: len(x) > 2)
corporate_filtered = corporate_filtered.drop('Symbol',axis = 1)

# convert the ratings to numerical values
ratings = corporate_filtered['Rating'].unique()
values = [3,4,2,5,6,7,10,8,1,9]
corporate_filtered['Rating'].replace(ratings,values,inplace=True)
corporate_filtered_np = corporate_filtered.to_numpy()
train,test = split_train_test(corporate_filtered_np)
# split into ratings and features
test_ratings, test_features = split_rating_features(test)
train_ratings, train_features = split_rating_features(train)
# MLP
model3 = MLPClassifier(hidden_layer_sizes=(4,6),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)
# fit the data
model3.fit(train_features,train_ratings)
# make predictions on the test set
preds=model3.predict(test_features)

# test the type of the test set
def test_test_split_type():
    assert type(test) == numpy.ndarray

# test the value of the mse calculation
def test_mse_value():
    mse = classification_mse(test_ratings, preds)
    assert mse >=0 and mse <= 1

# test the type of the mse calculation
def test_mse_type():
    mse = classification_mse(test_ratings, preds)
    assert type(mse) == float

# test the rating/feature split shape
def test_split_raing_features_shape():
    assert test_features.shape[1] == 25 and train_features.shape[1] == 25