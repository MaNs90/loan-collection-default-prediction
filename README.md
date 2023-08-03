# loan-collection-default-prediction
This model aims to output the probability of whether a user will default on his next installment so that loan officers can optimize their collection.

The model relies on 3 months worth of main data and 12 months of meta data to avoid data drift. It uses XGBoost to predict the probabilities but not before the data is segmented into three different segments with each its own model.
