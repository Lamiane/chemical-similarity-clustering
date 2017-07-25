#!/usr/bin/env python
# now the line above is super important!

# define single iteration of cross-validation
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import logging
logging.basicConfig()
from drgmum.toolkit import burrito
from chesicl.data_loading.exp_data_loader import load_data
from chesicl.experiment.utils import string_enhancer, serialise_confusion_matrix
from CONFIG import scoring_function

def get_hparams(max_clusters):
    n_clusters = list(range(1, max_clusters+1))
    init =['k-means++', 'random']
    algorithm = ['full', 'elkan']
    
    kmeans_hiperparameters = [{'n_clusters':nc, 'init':init_, 'algorithm':alg, 'n_init':1, 'max_iter':300, 'n_jobs':1} 
                       for nc in n_clusters for init_ in init for alg in algorithm]
    
    return kmeans_hiperparameters

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
    return {'result': scores,
           # 'confusion': serialise_confusion_matrix(confusion_matrix(y_train, predictions)),
           'score_name': string_enhancer(str(scoring_function))}

if __name__ == "__main__":
    burrito.print_as_json(burrito.wrap(cv_iteration, save_results=True))
