# Preprocess data
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FunctionTransformer, Pipeline

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
            ("extract_text", FunctionTransformer(lambda x_NC: x_NC["text"])),
            ("featurize", TfidfVectorizer()),
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
    print(
        "best auroc: ",
        # np.round(grid.cv_results_["mean_test_roc_auc"][grid.best_index_], 3),
        grid.cv_results_["mean_test_roc_auc"][grid.best_index_],
    )
    # print("best params: ", round_dict_vals(grid.best_params_))
    print("best params: ", grid.best_params_)
    print(
        "vocab size: ",
        len(grid.best_estimator_.named_steps["featurize"].vocabulary_),
    )
    best_model = grid.best_estimator_.named_steps["classify"]
    print(best_model)


C_GRID_VCOARSE = np.logspace(-4, 5, 10)
C_GRID_COARSE = np.logspace(-4, 5, 19)
C_GRID_MEDIUM = np.logspace(0, 1.5, 10)
C_GRID_FINE = np.logspace(0, 1, 21)
C_GRID = C_GRID_MEDIUM


def save_model(pipeline):
    featurizer = pipeline.steps[1]
    classifier = pipeline.steps[2][1]
    with open("problem2_featurizer.pickle", "wb") as f:
        pickle.dump(featurizer, f)
    with open("problem2_classifier.pickle", "wb") as f:
        pickle.dump(classifier, f)


PARAM_GRID = {
    "extract_text": [
        FunctionTransformer(text_col),
        FunctionTransformer(num_to_sp),
        FunctionTransformer(num_del),
    ],
    "featurize__token_pattern": [
        # r"(?u)\b\w+\b",
        r"(?u)\b\w\w+\b",
        r"(?u)\b\w\w\w+\b",
        # r"(?u)\b\w\w\w\w+\b",
    ],
    # "featurize__strip_accents": ["unicode", None],
    "featurize__strip_accents": ["unicode"],
    # "featurize__lowercase": [True, False],
    "featurize__lowercase": [True],
    # "classify__C": C_GRID_VCOARSE,
    "classify__C": np.logspace(-1, 3, 9),
    # "classify__l1_ratio": np.linspace(0.0, 1.0, 11),
    "classify__penalty": ["l2", "l1"],
}


def main():
    x_train_NC, y_train_N = load_data()
    grid = grid_search(x_train_NC, y_train_N, param_grid=PARAM_GRID)
    output_grid_result(grid)
    # save_model(grid.best_estimator_)


if __name__ == "__main__":
    main()
