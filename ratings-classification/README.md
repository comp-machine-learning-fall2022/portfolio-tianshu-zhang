# Credit Rating Classification
## Supervised Machine Learning with kNN, Decision Tree, and Random Forest

Corporations' credit ratings are often based on the company's  liquidity levels, profitability levels, debt levels, operations conditions, and cash flow properties. Using the corporate credit ratings data
set of 2029 companies, this project is interested in classifying the credit ratings of the companies within the data set given their financial performances. I'm also interested in the comparison between different
classification algorithms and compare their accuracy in the context of this credit rating data.

For this project, I implemented three classification algorithms from the sklearn package, k-Nearest-Neighbor, Decision Tree, and Random Forest classifiers. For each of the classification algorithms, I split the data into
90\% training and 10% testing and conducted 10-fold cross-validation on the training data. The classification mean squared error is calculated for each fold within the cross-validation as the measurement of accuracy of 
classification. The lower the 10-fold cross-validation error, the more accurate the model. For the kNN classification model, I reduced the data to 2D in order to visualize the classification result. The result comparing the
four algorithms showed that random forest is the best model of the four in terms of accuracy. Fitting the random forest model on the training data, the model is able to correctly classify around 42\% of the credit ratings of the 
companies within the test set. This result is still far from perfect given the fact that the source data is imbalanced and relatively small in terms of size. Future explorations could implement different classification algorithms
 such as SVM models with different kernels and fit the model on larger, more balanced data sets.

For a complete walk-through of this project, please refer to `Credit Rating Classification.ipynb`.