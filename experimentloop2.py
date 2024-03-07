# Preprocess data
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder

from preprocessing import num_del, num_to_sp, text_col

# Load data
# from load_train_data import overview_data
DATA_PATH = "./data_reviews"


def load_data(data_path=DATA_PATH):
    x_train_NC = pd.read_csv(os.path.join(DATA_PATH, "x_train.csv"))
    y_train_N = pd.read_csv(os.path.join(DATA_PATH, "y_train.csv"))
    # pandas reads y_train_N as a (1,N) column, but most functions just expect (N)
    return x_train_NC, y_train_N.squeeze()


def pipeline():
    return Pipeline(
        [
            (
                "featurize",
                ColumnTransformer(
                    [
                        ("review_bow", TfidfVectorizer(), 1),
                        (
                            "website_onehot",
                            OneHotEncoder(categories=[["amazon", "imdb", "yelp"]]),
                            [0],
                        ),
                    ]
                ),
            ),
            (
                "classify",
                LogisticRegression(max_iter=2000, solver="lbfgs"),
            ),
        ]
    )


def grid_search(x_NC, y_N, param_grid, return_train_score=False):
    grid = GridSearchCV(
        pipeline(),
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        return_train_score=return_train_score,
    )

    grid.fit(x_NC, y_N)
    return grid


def round_if_float(f, places=3):
    if isinstance(f, float):
        return np.round(f, places)
    return f


def round_dict_vals(d):
    return {k: round_if_float(v) for k, v in d.items()}


def output_grid_result(grid, name=None):
    if name:
        print(name)
    print("best auroc: ", grid.best_score_)
    print("best params: ", grid.best_params_)
    print(
        "vocab size: ",
        len(
            grid.best_estimator_.named_steps["featurize"]
            .named_transformers_["review_bow"]
            .vocabulary_
        ),
    )
    best_model = grid.best_estimator_.named_steps["classify"]
    print(best_model)


C_GRID_VCOARSE = np.logspace(-4, 5, 10)
C_GRID_COARSE = np.logspace(-4, 5, 19)
C_GRID_MEDIUM = np.logspace(0, 1.5, 10)
C_GRID_FINE = np.logspace(0, 1, 21)
C_GRID = C_GRID_MEDIUM


PARAM_GRID = {
    "featurize__review_bow__token_pattern": [
        r"(?u)\b\w+\b",
        r"(?u)\b\w\w+\b",
        r"(?u)\b\w\w\w+\b",
        # r"(?u)\b\w\w\w\w+\b",
    ],
    "featurize__review_bow__strip_accents": ["unicode", None],
    "featurize__review_bow__lowercase": [True, False],
    "featurize__review_bow__min_df": [1, 2, 3],
    "featurize__review_bow__max_df": np.logspace(-2, 0, 11),
    "classify__C": np.logspace(-2, 2, 5),
}


def main():
    x_train_NC, y_train_N = load_data()
    grid = grid_search(x_train_NC, y_train_N, param_grid=PARAM_GRID)
    output_grid_result(grid)
    # save_model(grid.best_estimator_)


if __name__ == "__main__":
    main()
