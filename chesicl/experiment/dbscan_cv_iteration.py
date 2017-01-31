#!/usr/bin/env python
# now the line above is super important!

# define single iteration of cross-validation
from sklearn.cluster import DBSCAN
import numpy as np
import os
import logging
logging.basicConfig()

from drgmum.toolkit import burrito

from chesicl.data_loading.exp_data_loader import load_data
from experiment import scoring_function, random_state

# default parameters are must-be!
def cv_iteration(n_jobs=4, eps=1, min_samples=30, metric='euclidean', algorithm='brute', leaf_size=30, p=2):
    X, y_train = load_data()
    scores = []
    model = KMeans(n_jobs=n_jobs, eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size, p=p)
    model.fit(X)
    predictions = model.predict(X)
    scores.append(scoring_function(y_train, predictions))
    return {'result': scores}

if __name__ == "__main__":
     burrito.print_as_json(burrito.wrap(cv_iteration, save_results=True))
