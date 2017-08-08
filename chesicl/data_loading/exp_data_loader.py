# imports
import os
import numpy as np
import cPickle as pkl
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from itertools import product
from bidict import bidict
import logging
logging.basicConfig()
import drgmum
from drgmum.toolkit.dataset import SmartDataset
from drgmum.chemdata.fingerprints import molprint2d_count_fingerprinter
from drgmum.chemdata.chembl.sqlite_queries import get_smiles
from drgmum.chemdata.utils.pandas_utils import two_level_series_to_csr
from load_constraints import get_chembls, load_similarity_matrices, get_mapping

random_state = 666
data_folder = '/var/data/users/local/pocha/data'
chembl21db = os.path.join(data_folder, 'DRGMUM/chembl_21.db')
chembl_ids_file = os.path.join(data_folder, 'SCFP/Random_compounds_100.sdf')
similarity_matrix_file = os.path.join(data_folder, 'SCFP/Similarity150Dawid.csv')
folder_with_pairs = os.path.join(data_folder, 'SCFP/pairs')
calculated_data = "666experiment_data_doNOTremove4832.pkl"

def load_data():
    # maybe the data can be load from disk
    # TODO: what would happen in case of multiple threads?...
    data_file_path = os.path.join(data_folder, calculated_data)
    if os.path.isfile(data_file_path):
        with open(data_file_path, 'r') as f:
            X, y_train, y_test = pkl.load(f)
        return X, y_train, y_test
        
    # load data

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

    # 6. split constraints into folds
    y_train, y_test = train_test_split(bin_sim_list, train_size=0.8, random_state=random_state)
    # 343 - train, 86 - test 

    # 7. just for convenience
    X = sprase_fp.toarray()
    
    with open(data_file_path, 'w') as f:
        pkl.dump((X, y_train, y_test), f)

    return X, y_train, y_test


def load_data_random_labels(seed):
    X, y_train, y_test = load_data()
    np.random.seed(seed)
    return X, np.random.shuffle(y_train), np.random.shuffle(y_test)
