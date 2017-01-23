#!/usr/bin/env python
# now the line above is super important!

# define single iteration of cross-validation
from drgmum.toolkit import burrito
# imports
import drgmum
from drgmum.toolkit.dataset import SmartDataset
from drgmum.chemdata.fingerprints import molprint2d_count_fingerprinter
from drgmum.chemdata.chembl.sqlite_queries import get_smiles
from drgmum.chemdata.utils.pandas_utils import two_level_series_to_csr
import logging
logging.basicConfig()
import os
data_folder = '/var/data/users/local/pocha/data'
chembl21db = os.path.join(data_folder, 'DRGMUM/chembl_21.db')
chembl_ids_file = os.path.join(data_folder, 'SCFP/Random_compounds_100.sdf')
similarity_matrix_file = os.path.join(data_folder, 'SCFP/Similarity150Dawid.csv')
folder_with_pairs = os.path.join(data_folder, 'SCFP/pairs')
from load_constraints import get_chembls, load_similarity_matrices, get_mapping
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
random_state = 666
from bidict import bidict
import numpy as np
from itertools import product

def load_data():
    # 1. load chembl IDs
    chembl_ids = get_chembls(chembl_ids_file)
    smiles = get_smiles(chembl21db, chembl_ids)

    # 2. load fingerprints
    a = molprint2d_count_fingerprinter(smiles) # not important
    sprase_fp, row_labels, col_labels = two_level_series_to_csr(a)

    # 3. load constraints
    bin_sim, scale_sim, mapping_idx_chembl = load_similarity_matrices(similarity_matrix_file, chembl_ids_file, folder_with_pairs)
    bin_sim_list = zip(zip(*bin_sim.nonzero()), bin_sim.data)
    scale_sim_list = zip(zip(*scale_sim.nonzero()), scale_sim.data)

    # 4. remove contraints that are duplicate (sim(x,y)=sim(y,x)) or
    bin_sim_list = [x for x in bin_sim_list if x[0][0]<x[0][1]]
    scale_sim_list = [x for x in scale_sim_list if x[0][0]<x[0][1]]

    # 5. chcemy by mapowanie indeks-chembl dla sparse_fp i dla bin/scale_sim bylo takie samo
    constraints_mapping = get_mapping(all_compounds_file=chembl_ids_file) # digit: chembl_id
    fp_mapping = bidict(zip(row_labels, range(len(row_labels))))

    constraints_2_fp_mapping = bidict([])
    for chembl_id in row_labels:
        constraints_2_fp_mapping[constraints_mapping.inv[chembl_id]] = fp_mapping[chembl_id]

    bin_sim_list = [((constraints_2_fp_mapping[x[0][0]], constraints_2_fp_mapping[x[0][1]]), x[1]) for x in bin_sim_list]
    scale_sim_list = [((constraints_2_fp_mapping[x[0][0]], constraints_2_fp_mapping[x[0][1]]), x[1]) for x in scale_sim_list]

    # 6. podzial wiezow na foldy
    y_train, y_test = train_test_split(bin_sim_list, train_size=0.8, random_state=random_state)
    # 343 - train, 86 - test 

    # 7. just for convenience
    X = sprase_fp.toarray()
    return X, y_train
    
# define score
def my_binary_score(labels_true, labels_pred):
    # labels true: ((x_i, x_j), sim(x_i, x_j))
    # labels_pred: list
    true, pred = np.array(labels_true), np.array(labels_pred)
    diffs = 0
    for ((idx_1, idx_2), sim_true) in labels_true:
        if labels_pred[idx_1] == labels_pred[idx_2]:
            diffs = diffs + (np.absolute(sim_true-1.))
        else:
            diffs = diffs + (np.absolute(sim_true-0.))
    return diffs/len(labels_true)

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
