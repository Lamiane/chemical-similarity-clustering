#!/usr/bin/env python
# now the line above is super important!

# define single iteration of cross-validation
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from bidict import bidict
import numpy as np
from itertools import product
import os
import logging
logging.basicConfig()

from drgmum.toolkit import burrito
import drgmum
from drgmum.toolkit.dataset import SmartDataset
from drgmum.chemdata.fingerprints import molprint2d_count_fingerprinter
from drgmum.chemdata.chembl.sqlite_queries import get_smiles
from drgmum.chemdata.utils.pandas_utils import two_level_series_to_csr

from load_constraints import get_chembls, load_similarity_matrices, get_mapping
from chesicl.data_loading.exp_data_loader import load_data
from scorer import my_binary_score

random_state = 666
data_folder = '/var/data/users/local/pocha/data'
chembl21db = os.path.join(data_folder, 'DRGMUM/chembl_21.db')
chembl_ids_file = os.path.join(data_folder, 'SCFP/Random_compounds_100.sdf')
similarity_matrix_file = os.path.join(data_folder, 'SCFP/Similarity150Dawid.csv')
folder_with_pairs = os.path.join(data_folder, 'SCFP/pairs')

# define models and hiperparameters
n_clusters = list(range(1, 101))
init =['k-means++', 'random']
n_init = [1]
algorithm = ['full', 'elkan']
hiperparameters = [{'n_clusters':nc, 'init':init_, 'n_init':ninit, 'algorithm':alg} 
                   for nc in n_clusters for init_ in init for ninit in n_init for alg in algorithm]

# default parameters are must-be!
def cv_iteration(n_clusters=1, n_init=1, init='random', algorithm='elkan'):
    X, y_train = load_data()
    scores = []
    random_states = (666, 69, 7, 13, 1337)
    for i in random_states:
        model = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, algorithm=algorithm, random_state=i)
        model.fit(X)
        predictions = model.predict(X)
        scores.append(my_binary_score(y_train, predictions))
    return {'result': scores}

if __name__ == "__main__":
     burrito.print_as_json(burrito.wrap(cv_iteration, save_results=True))
