# How to proceed through data cleaning (inner module in outer loop)

# Preprocess data
import os

import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate

# from sklearn.dummy import DummyClassifier

# Load data
# from load_train_data import overview_data
DATA_PATH = "./data_reviews"


def load_data(data_path=DATA_PATH):
    x_train_NC = pd.read_csv(os.path.join(DATA_PATH, "x_train.csv"))
    y_train_N = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))
    # pandas reads y_train_N as a (1,N) column, but most functions just expect (N)
    return x_train_NC, y_train_N.squeeze()


# add fxn contract
def featurize(x_NC):
    vectorizer = CountVectorizer()
    corpus = x_NC.loc[:, "text"]
    # n rows, v features (vocab)
    return vectorizer.fit_transform(corpus)  # may want to add vectorizer later


# overview_data(x_train_nf, y_train_n)
def cross_validation_auroc(x_NF, y_N):
    model = LogisticRegression()

    # produce auroc scores by 5-fold cross-validation on (x_NF,y_N)
    scores = cross_validate(model, x_NF, y_N, cv=5, scoring="roc_auc")

    return np.mean(scores["test_score"])


x_train_NC, y_train_N = load_data()
features_NF = featurize(x_train_NC)
avg_cv_scores = cross_validation_auroc(features_NF, y_train_N)
print("avg:", np.round(avg_cv_scores, 3))


# model.fit(input_vectors_nv, y_train_n)
# baseline_test_scores = cross_validate(
#    DummyClassifier(strategy="constant", constant=0),
#    input_vectors_nv,
#    y_train_n,
#    cv=5,
#    scoring="roc_auc",
# )
# average_baseline_cv_scores = np.mean(baseline_test_scores["test_score"])
# print("baseline: ", average_baseline_cv_scores)

# Cross validation
# def cross_validate(model): ...


# split into folds

# for each fold, fit and evaluate


# Call CV function
# auroc = cross_validate(model)


# TODO Refine with hyperparameters

# TODO Evaluate model
