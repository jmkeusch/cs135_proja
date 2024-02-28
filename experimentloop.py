# How to proceed through data cleaning (inner module in outer loop)

# TODO Preprocess data
import os

import pandas as pd

# import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate


# Load data
# from load_train_data import overview_data
DATA_PATH = "./data_reviews"

x_train_nf = pd.read_csv(os.path.join(DATA_PATH, "x_train.csv"))
y_train_n = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))
y_train_n = y_train_n.to_numpy().squeeze()

# overview_data(x_train_nf, y_train_n)

vectorizer = CountVectorizer()
corpus = x_train_nf.loc[:, "text"]
# n rows, v features (vocab)
input_vectors_nv = vectorizer.fit_transform(corpus)
input_vectors_nv.shape

#  Fit model
model = LogisticRegression()
# model.fit(input_vectors_nv, y_train_n)

test_scores = cross_validate(
    model, input_vectors_nv, y_train_n, cv=5, scoring="roc_auc"
)

print(test_scores)


# Cross validation
# def cross_validate(model): ...


# split into folds

# for each fold, fit and evaluate


# Call CV function
# auroc = cross_validate(model)


# TODO Refine with hyperparameters

# TODO Evaluate model
