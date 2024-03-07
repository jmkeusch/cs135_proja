# This file is suppose to provide an interface between your implementation and the autograder.
# In reality, the autograder should be a production system. This file provide an interface for
# the system to call your classifier.

# Ideally you could bundle your feature extractor and classifier in a single python function,
# which takes a raw instance (a list of two strings) and predict a probability.

# Here we use a simpler interface and provide the feature extractor and the classifer separately.
# For Problem 2, you are supposed to provide
# * a feature extraction function `extract_awesome_features`, and
# * a sklearn classifier, `classifier2`, whose `predict_proba` will be called.
# * your team name

# These two python objects will be imported by the `test_classifier_before_submission` autograder.

import pickle

from sklearn.pipeline import FunctionTransformer, Pipeline

from preprocessing import num_to_sp

with open("problem1_featurizer.pickle", "rb") as f:
    ...
    featurizer_step = pickle.load(f)

extract_awesome_features = Pipeline(
    [("extract_text", FunctionTransformer(num_to_sp)), featurizer_step]
)

with open("problem1_classifier.pickle", "rb") as f:
    classifier2 = pickle.load(f)

teamname = "Model Behavior"
