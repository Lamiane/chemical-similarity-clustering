#!/usr/bin/env python
# now the line above is super important!

# define single iteration of cross-validation
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import logging
logging.basicConfig()
from chesicl.toolkit import burrito
from chesicl.data_loading.exp_data_loader import load_data
from chesicl.experiment.utils import string_enhancer, serialise_confusion_matrix
from CONFIG import scoring_function

def get_hparams(max_clusters):
    n_clusters = list(range(1, max_clusters+1))    
    affinity = ["cosine", "euclidean"]
    linkage = ["average", "complete", "ward"]
    
    agglomerative_hiperparameters = [{'n_clusters':nc, 'affinity':affinity_, 'linkage':link} 
                       for nc in n_clusters for affinity_ in affinity for link in linkage
                              if link is not 'ward' or affinity is 'euclidean']
    
    return agglomerative_hiperparameters

# default parameters are must-be!
def cv_iteration(n_clusters=1, affinity='euclidean', linkage='ward'):
    X, y_train, _ = load_data()
    scores = []
    cms = []  # confusion matrices
    cluster_sizes = []
    random_states = (666, 69, 7, 13, 1337)
    for i in random_states:
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        predictions = model.fit_predict(X)
        score, confusion_matrix = scoring_function(y_train, predictions)
        scores.append(score)
        cms.append(serialise_confusion_matrix(confusion_matrix))
        cluster_sizes.append(serialise_confusion_matrix(np.unique(predictions, return_counts=True)))
                  
    return {'result': scores,
           'confusion_matrices': eval(str(cms)),
           'score_name': string_enhancer(str(scoring_function)),
           'cluster_sizes': eval(str(cluster_sizes))}

if __name__ == "__main__":
    burrito.print_as_json(burrito.wrap(cv_iteration, save_results=True))
