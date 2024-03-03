# How to proceed through data cleaning (inner module in outer loop)

# Preprocess data
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import FunctionTransformer, Pipeline

# from sklearn.dummy import DummyClassifier

# Load data
# from load_train_data import overview_data
DATA_PATH = "./data_reviews"


def load_data(data_path=DATA_PATH):
    x_train_NC = pd.read_csv(os.path.join(DATA_PATH, "x_train.csv"))
    y_train_N = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))
    # pandas reads y_train_N as a (1,N) column, but most functions just expect (N)
    return x_train_NC, y_train_N.squeeze()


def pipeline():
    text_col_extractor = FunctionTransformer(lambda x_NC: x_NC.loc[:, "text"])
    vectorizer = CountVectorizer()
    classifier = LogisticRegression(max_iter=400)
    return Pipeline(
        [
            ("extract_text", text_col_extractor),
            ("featurize", vectorizer),
            ("classify", classifier),
        ]
    )


def grid_search(x_NC, y_N, param_grid):
    grid = GridSearchCV(pipeline(), param_grid=param_grid, scoring="roc_auc", n_jobs=-1)

    grid.fit(x_NC, y_N)
    return grid


def round_if_float(f, places=3):
    if isinstance(f, float):
        return np.round(f, places)
    return f


def round_dict_vals(d):
    return {k: round_if_float(v) for k, v in d.items()}


def output_grid_result(name, grid):
    print(name)
    print("best score: ", np.round(grid.best_score_, 3))
    print("best params: ", round_dict_vals(grid.best_params_))
    print(
        "vocab size: ",
        len(grid.best_estimator_.named_steps["featurize"].vocabulary_),
    )
    best_model = grid.best_estimator_.named_steps["classify"]
    print(best_model)


C_GRID = np.logspace(-9, 6, 31)


def main():
    x_train_NC, y_train_N = load_data()
    param_grids = {
        "baseline": {"classify": [DummyClassifier(strategy="constant", constant=0)]},
        "count_default": {
            "featurize": [CountVectorizer()],
            "classify__C": C_GRID,
        },
        "count_top_2000_words": {
            "featurize": [CountVectorizer(max_features=2000)],
            "classify__C": C_GRID,
        },
        "tfidf_default": {
            "featurize": [TfidfVectorizer()],
            "classify__C": C_GRID,
        },
    }
    for name, grid in param_grids.items():
        grid = grid_search(x_train_NC, y_train_N, param_grid=grid)
        print("\n\n")
        output_grid_result(name, grid)


if __name__ == "__main__":
    main()

# TODO Refine with hyperparameters

# TODO Evaluate model
