# Tianshu Zhang's Machine Learning Portfolio

## Introduction

Hi my name is Tianshu Zhang and welcome to my machine learning portfolio. I'm a graduating senior at Smith College studying 
Quantitative Economics and Mathematical Statistics with a concentration in Global Finance. This portfolio is a summary of the skills
I learned in my machine learning course CSC294 Computation Machine Learning under Dr. Katherine Kinnaird. Besides being the final project for the course,
this portfolio is also designed to document my machine learning projects and demonstrate my machine learning skills for potential
employers. 

Some of the skills that I choose to demonstrate in this portfolio are:
1. Data wrangling and visualization (PCA dimension reduction, `numpy`,`pandas`, `pyplot` visualization)
2. [Clustering](sector-clustering): k-means clustering, benchmarking, line-by-line profiling, `sklearn`
3. [Classification](ratings-classification): k-nearest-neighbors, decision trees, random forests, cross-validation, `sklearn`
4. [Deep Learning](deep-learning): Neural Networks, Multi-layer Perceptron Neural Networks (MLP), `sklearn`,`tensorflow`

## Data set

Throughout the portfolio, I explore a [credit rating data set](https://www.kaggle.com/datasets/agewerc/corporate-credit-rating) that contains
a list of 2029 credit ratings issued by major agencies such as Standard and Poors to big US firms (traded on NYSE or Nasdaq) from 2010 to 2016.
The data set is stored as `corporate_rating.csv`.

There are 30 features for every company of which 25 are financial indicators. They can be divided in:

- Liquidity Measurement Ratios: currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding
- Profitability Indicator Ratios: grossProfitMargin, operatingProfitMargin, pretaxProfitMargin, netProfitMargin, effectiveTaxRate, returnOnAssets, returnOnEquity, returnOnCapitalEmployed
- Debt Ratios: debtRatio, debtEquityRatio
- Operating Performance Ratios: assetTurnover
- Cash Flow Indicator Ratios: operatingCashFlowPerShare, freeCashFlowPerShare, cashPerShare, operatingCashFlowSalesRatio, freeCashFlowOperatingCashFlowRatio

The dataset is unbalanced, here is the frequency of ratings:
- AAA: 7
- AA: 89
- A: 398
- BBB: 671
- BB: 490
- B: 302
- CCC: 64
- CC: 5
- C: 2
- D: 1

## File structure

For each of the projects within the portfolio, there are three types of files:

- `README.md`: the readme file that provides an overview of the project
- `.py`,`test_ .py`: python files that contain the functions used in the project and the corresponding unit tests
- `.png`: screenshots of unit tests passing and other visualizations
