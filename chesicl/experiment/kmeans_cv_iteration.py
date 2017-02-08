#!/usr/bin/env python
# now the line above is super important!

# define single iteration of cross-validation
from sklearn.cluster import KMeans
import numpy as np
import os
import logging
logging.basicConfig()

from drgmum.toolkit import burrito

from chesicl.data_loading.exp_data_loader import load_data
from experiment import scoring_function

# default parameters are must-be!
def cv_iteration(n_clusters=1, n_init=1, init='random', algorithm='elkan', max_iter=300, n_jobs=1):
    X, y_train, _ = load_data()
    scores = []
    random_states = (666, 69, 7, 13, 1337)
    for i in random_states:
        model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, algorithm=algorithm, random_state=i, n_jobs=n_jobs)
        model.fit(X)
        predictions = model.predict(X)
        scores.append(scoring_function(y_train, predictions))
    return {'result': scores}

if __name__ == "__main__":
    burrito.print_as_json(burrito.wrap(cv_iteration, save_results=True))
