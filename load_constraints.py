"""
Set of functions for loading the dataset.
"""

import bidict
import os
import csv
import scipy.sparse
import numpy as np
import random
import matplotlib.pyplot as plt


def get_chembls(filename):
    """
    Extracting CHEMBLs from sdf file

    Parameters
    ----------
    filename: str
        Name of the sdf file

    Returns
    -------
    result : list
        Returns list containing all CHEMBLs as strings.
    """
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
    """
    Mapping CHEMBLs to integers so that they can be used as indices of an array

    Parameters
    ----------
    all_compounds_file: str
        Name of file containing all compounds

    Returns
    -------
    mapping : bidict
        Returns bidirectional map CHMEBL-index
    """
    mapping = bidict.bidict()
    all_chembls = get_chembls(all_compounds_file)
    mapping.update(dict(zip(xrange(len(all_chembls)), all_chembls)))
    return mapping


def get_all_files(path):
    """
    List all files in the directory

    Parameters
    ----------
    path: str
        path to directory in which function will look for files

    Returns
    -------
    : list
        Returns list containing paths to all files
    """
    return [os.path.join(path, filename) for filename in os.listdir(path)
            if os.path.isfile(os.path.join(path, filename))]


def find_non_unique_pairs(dict_idx_chemblchembl, file_numbers=False):
    """
    Find all non-unique pairs in data.

    Parameters
    ----------
    dict_idx_chemblchembl: dict
        d[number_from_file_name] = (chembl_1, chembl_2)

    file_numbers: True or False
        should d[file_number_1] = [file_number_2, file_number_3, ...],\
         where file_number_[123] contain same pair, be returned?

    Returns
    -------
    pairs_to_omit: set
        numbers from file names of those files which are redundant

    non_unique: dict
        d[file_number_1] = [file_number_2, file_number_3, ...], only if file_numbers is set to True
    """
    non_unique = {}
    # file_numbers = []
    for key in sorted(dict_idx_chemblchembl.keys()):
        chembl_i, chembl_j = dict_idx_chemblchembl[key]
        for another_key in sorted(dict_idx_chemblchembl.keys()):
            if key != another_key \
                    and chembl_i in dict_idx_chemblchembl[another_key]\
                    and chembl_j in dict_idx_chemblchembl[another_key]:
                # file_numbers.append((chembl_i, chembl_j))
                new_key = min(key, another_key)
                new_value = max(key, another_key)
                if new_key not in non_unique.keys():
                    non_unique[new_key] = [new_value]
                else:
                    if non_unique[new_key] is None:
                        print non_unique
                    non_unique[new_key].append(new_value)
    pairs_to_omit = set([item for sublist in non_unique.values() for item in sublist])
    if file_numbers:
        for key in non_unique.keys():
            non_unique[key] = set(non_unique[key])
        return pairs_to_omit, non_unique
    else:
        return pairs_to_omit


def load_similarity_matrices(similarity_matrix_file, all_compounds_file, folder_with_pairs):
    """
    Load similarity matrices

    Parameters
    ----------
    similarity_matrix_file: str
        Path to file containing similarity matrix

    all_compounds_file: str
        Path to file with all compounds

    folder_with_pairs: str
        Path to folder with files describing pairs

    Returns
    -------
    tuple containing:

    bin_similarity : scipy.sparse.csr_matrix
        Sparse matrix with binary score (-1 not similar, 1 similar)

    scale_similarity : scipy.sparse.csr_matrix
        Sparse matrix with similarity score in scale 1 (not similar) to 5 (similar)

    mapping_idx_chembl : bidict
        Bidirectional map CHMEBL-index
    """
    bin_similarity = []
    scale_similarity = []
    row_ind = []
    col_ind = []

    mapping_idx_chembl = get_mapping(all_compounds_file)
    n_compunds = len(mapping_idx_chembl)
    dict_idx_chemblchembl = dict([[int(''.join(c for c in filename if c.isdigit())), tuple(get_chembls(filename))]
                                  for filename in get_all_files(folder_with_pairs)])

    #######################################
    # # # ensure all pairs are unique # # #
    #######################################
    # TODO: now repeats are removed but maybe an average score should be derived?
    pairs_to_omit = find_non_unique_pairs(dict_idx_chemblchembl)
    print len(pairs_to_omit), 'pairs were omitted'

    n_omitted = 0
    with open(similarity_matrix_file, 'r') as csvfile:
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


def load_sabina(path):
    from pandas import DataFrame
    df = DataFrame.from_csv(path)

if __name__ == '__main__':
    bin_sim, scale_sim, mapping_idx_chembl = \
        load_similarity_matrices('Similarity.csv', 'Random_compounds_100.sdf', 'pairs')
    print 'bin_sim\n', bin_sim
    print '\nscale_sim\n', scale_sim
    print '\nmapping_idx_chembl\n', mapping_idx_chembl
