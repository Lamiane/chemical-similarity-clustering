import bidict
import os
import csv
import scipy.sparse
import numpy as np
import random
import matplotlib.pyplot as plt


def get_chembls(filename):
    result = []
    first_line = True
    with open(filename, 'r') as f:
        for line in f:
            if first_line:
                assert 'CHEMBL' in line
                result.append(line.strip())
                first_line = False
            elif '$$$$' in line:
                first_line = True
            else:
                pass
    return result


def get_mapping(all_compounds_file):
    mapping = bidict.bidict()
    all_chembls = get_chembls(all_compounds_file)
    mapping.update(dict(zip(xrange(len(all_chembls)), all_chembls)))

    return mapping


def get_all_files(path):
    return [os.path.join(path, filename) for filename in os.listdir(path)
            if os.path.isfile(os.path.join(path, filename))]


def load_similarity_matrices(all_compunds_file, folder_with_pairs):
    bin_similarity = []
    scale_similarity = []
    row_ind = []
    col_ind = []

    mapping_idx_chembl = get_mapping(all_compunds_file)
    n_compunds = len(mapping_idx_chembl)
    dict_idx_chemblchembl = \
    dict([[int(''.join(c for c in filename if c.isdigit())), tuple(get_chembls(filename))]
          for filename in get_all_files(folder_with_pairs)])

    ###################################
    # # # zapewnij unikalnosc par # # #
    ###################################
    non_unique = {}
    for key in sorted(dict_idx_chemblchembl.keys()):
        chembl_i, chembl_j = dict_idx_chemblchembl[key]
        for another_key in sorted(dict_idx_chemblchembl.keys()):
            if key != another_key \
                    and chembl_i in dict_idx_chemblchembl[another_key]\
                    and chembl_j in dict_idx_chemblchembl[another_key]:
                new_key = min(key, another_key)
                new_value = max(key, another_key)
                if new_key not in non_unique.keys():
                    non_unique[new_key] = [new_value]
                else:
                    if non_unique[new_key] is None:
                        print non_unique
                    non_unique[new_key].append(new_value)
    pairs_to_omit = set([item for sublist in non_unique.values() for item in sublist])

    print len(pairs_to_omit), 'pairs were omitted'
    ######################
    # # # zapewniono # # #
    ######################

    n_omitted = 0
    with open('Similarity.csv', 'r') as csvfile:
        for pair_number, bin_sim, scale_sim in csv.reader(csvfile, delimiter=','):
            if int(pair_number) not in pairs_to_omit:
                chembl_i, chembl_j = dict_idx_chemblchembl[int(pair_number)]
                chembl_i_idx, chembl_j_idx = mapping_idx_chembl.inv[chembl_i], mapping_idx_chembl.inv[chembl_j]
                row_ind.extend([chembl_i_idx, chembl_j_idx])
                col_ind.extend([chembl_j_idx, chembl_i_idx])
                bin_similarity.extend([int(bin_sim), int(bin_sim)])
                scale_similarity.extend([int(scale_sim), int(scale_sim)])
            else:
                n_omitted += 1

    assert (len(dict_idx_chemblchembl)-n_omitted)*2 == len(bin_similarity)
    assert len(bin_similarity) == len(scale_similarity)
    assert len(scale_similarity) == len(row_ind)
    assert len(row_ind) == len(col_ind)

    # we want bin similarities to be -1, 1 not 0, 1
    bin_similarity = [(-1)**(1-x) for x in bin_similarity]
    assert 0 not in scale_similarity
    assert 0 not in bin_similarity

    scale_similarity = scipy.sparse.csr_matrix((scale_similarity, (row_ind, col_ind)), shape=(n_compunds, n_compunds))
    bin_similarity = scipy.sparse.csr_matrix((bin_similarity,   (row_ind, col_ind)), shape=(n_compunds, n_compunds))

    assert scale_similarity.nnz == len(row_ind)
    assert bin_similarity.nnz == len(row_ind)

    # a compound is always similar to itself
    scale_similarity.setdiag(5*np.ones((scale_similarity.shape[0])))
    bin_similarity.setdiag(np.ones((bin_similarity.shape[0])))

    assert np.all(scale_similarity.todense() == np.transpose(scale_similarity.todense()))
    assert scale_similarity.shape == (n_compunds, n_compunds)
    assert np.all(bin_similarity.todense() == np.transpose(bin_similarity.todense()))
    assert bin_similarity.shape == (n_compunds, n_compunds)

    assert scale_similarity.nnz == bin_similarity.nnz
    assert np.all(scale_similarity.indices == bin_similarity.indices)
    assert np.all(scale_similarity.nonzero()[0] == bin_similarity.nonzero()[0])
    assert np.all(scale_similarity.nonzero()[1] == bin_similarity.nonzero()[1])

    return bin_similarity, scale_similarity, mapping_idx_chembl
