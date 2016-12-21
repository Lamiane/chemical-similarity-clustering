"""
Set of functions to find the dataset's best splits.
"""

import bidict
import os
import csv
import scipy.sparse
import numpy as np
import random
import matplotlib.pyplot as plt
import load_constraints

"""
some constants useful throughout the project
"""
# constants
n_sets = 10
possible_compounds = 'possible compounds'
pairs_contained = 'pairs contained'
probability = 'probability'
compounds_contained = 'compounds contained'
size = 'size'


def split_loss(folds, n_omitted):
    # TODO docs
    """
    Loading similarity matrices

    Parameters
    ----------
    folds: list of dictionaries
        List with dictionaries describing each fold.

    n_omitted: int
        Number of pairs not included in any of the folds.

    Returns
    -------
    tuple containing:

    loss : int
        the final loss (0 - perfect loss, no upper bound)

    variance : int
        semi-variance over the folds (takes into account desirable relative folds size)
    """
    # we need low variance and small number of pairs that were not included
    assert isinstance(folds, list), 'folds should be a non-empty list of dictionaries'
    assert len(folds) > 0, 'folds should be a non-empty list of dictionaries'
    # TODO wanna remake the assert below: for f in folds isinstance(f, dict)
    assert isinstance(folds[0], dict), 'folds should be a non-empty list of dictionaries'
    variance = 0
    if pairs_contained in folds[0].keys():
        variance = np.var([float(len(fold[pairs_contained]))/(fold[probability][1]-fold[probability][0])
                           for fold in folds])
    elif size in folds[0].keys():
        variance = np.var([float(fold[size])/(fold[probability][1]-fold[probability][0])
                           for fold in folds])
    else:
        raise KeyError('either `'+size+'` or `'+pairs_contained+'` must be in folds')
    loss = (5./8)*variance + n_omitted
    return loss, variance


def agnieszka_splits(similarity_matrix_file, all_compounds_file, folder_with_pairs):
    """
    Function that looks through random splits and picks best of them.

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
    best_split : list
        List of n_sets best splits. Split itself is a list of dictionaries describing folds.
        Each dictionary describes one fold and contains:
        possible_compounds: list of compounds that can be put into this fold
        pairs_contained: list of pairs which similarity score we know that are contained in this fold
        probability: probability of drawing the fold (proportional to its size)
    """
    best_splits = []
    best_loss = 10**4
    losses = []
    variances = []
    n_ommited_pairs = []
    try:
        while True:  # many times
            # TODO: inicjalizacja zmiennych moze tylko raz?
            bin_sim, _, mapping_idx_chembl = \
                load_constraints.load_similarity_matrices(similarity_matrix_file, all_compounds_file, folder_with_pairs)
            bin_sim.setdiag(np.zeros((bin_sim.shape[0])))  # pairs (i,i) are not interesting for us
            # TODO ad msc.1. pairs = [p for p in pairs if p[0]<p[1]
            # TODO cd. wtedy nawet diagonali nie trzeba zerowac, bo ostry warunek
            pairs = zip(bin_sim.nonzero()[0], bin_sim.nonzero()[1])

            # note do TODO: shuffle powinnien byc wykonywany przy kazdym obiedu petli
            random.shuffle(pairs)
            folds = [{possible_compounds: list(mapping_idx_chembl.keys()), pairs_contained: []} for _ in xrange(2)]
            folds[0][probability], folds[1][probability] = (0.0, 0.9), (0.9, 1.0)
            failures = 0
            max_failures = len(pairs)
            try:
                while failures < max_failures:
                    i, j = pairs.pop()  # popping random pair
                    # msc.1.
                    if i > j:
                        continue  # saving time

                    x = np.random.rand()
                    for fold in folds:  # iterating over folds to find the chosen one
                        if fold[probability][0] <= x < fold[probability][1]:  # if the fold was chosen
                            # if pair might go inside
                            if i in fold[possible_compounds] and j in fold[possible_compounds]:
                                failures = 0
                                fold[pairs_contained].append((i, j))

                                # remove compounds i, j from possible_compounds in each fold
                                for c in folds:
                                    if i in c[possible_compounds]:
                                        c[possible_compounds].remove(i)
                                    if j in c[possible_compounds]:
                                        c[possible_compounds].remove(j)
                                fold[possible_compounds].extend([i, j])
                            else:  # the fold was chosen but we cannot fit the pair there
                                failures += 1
                                # TODO: dlaczego na poczatku? przeciez to moze potencjalnie blokowac...
                                pairs.insert(0, (i, j))  # pair goes back to the poll
                            break  # we've found the fold chosen for this pair

            except IndexError:  # pairs is empty
                print 'index error'
                pass
            finally:
                # out of while, so either all pairs are included or number of failures was too great
                omitted_pairs = len(pairs)
                loss, variance = split_loss(folds, omitted_pairs)

                print 'loss', loss
                print 'omitted_pairs', omitted_pairs
                if loss < best_loss:
                    print '\n'

                    best_loss = loss
                    best_splits.append(folds)
                    if len(best_splits) > n_sets:
                        best_splits = best_splits[-n_sets:]
                    losses.append(best_loss)
                    variances.append(variance)
                    n_ommited_pairs.append(omitted_pairs)
                    plt.plot(range(len(losses)), losses, c='r')
                    plt.plot(range(len(variances)), variances, c='g')
                    plt.plot(range(len(n_ommited_pairs)), n_ommited_pairs, c='b')
                    plt.title('best loss over time')
                    plt.show()
                    print "loss", loss, 'variance', variance, 'omitted_pairs', omitted_pairs
                    print 'folds\' sizes', [len(fold[pairs_contained]) for fold in folds]
                    print 'omitted_pairs', omitted_pairs
                    print "#BEST SPLIT\n", best_splits[-1]

                print '_________________________________________________\n'
    except KeyboardInterrupt:
        print 'kotek'

    return best_splits


def staszek_splits(similarity_matrix_file, all_compounds_file, folder_with_pairs):
    """
    Function that looks through random splits and picks best of them.

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
    best_splits : list
        list of n_sets best splits.
        Splitting itself is a list of dictionaries describing folds. Each dictionary describes one fold and contains:
        probability: probability of drawing the fold (proportional to its size)
        compounds_contained: compounds consisting the fold
    """
    best_splits = []
    best_loss = 10**4
    losses = []
    variances = []
    n_omitted_pairs = []

    try:
        while True:  # many times
            # TODO inicjalizacja moze byc przed forem
            bin_sim, _, mapping_idx_chembl = \
                load_constraints.load_similarity_matrices(similarity_matrix_file, all_compounds_file, folder_with_pairs)
            bin_sim.setdiag(np.zeros((bin_sim.shape[0])))  # pairs (i,i) are not interesting for us
            # TODO ad msc.2. paris = [p for p in pairs if p[0] < p[1] ]
            pairs = zip(bin_sim.nonzero()[0], bin_sim.nonzero()[1])

            # ad TODO shuffle musi byc w forze
            random.shuffle(pairs)
            folds = [{compounds_contained: [], size:0} for i in xrange(2)]
            folds[0][probability], folds[1][probability] = (0.0, 0.9), (0.9, 1.0)

            omitted_pairs = 0

            # TODO tu chyba nie powinno byc len od pairs tylko od numerow compoundow
            for compound_index in xrange(0, len(pairs)):
                x = np.random.rand()
                for fold in folds:  # iterating over folds to find the chosen one
                        if fold[probability][0] <= x < fold[probability][1]:  # if the fold was chosen
                            fold[compounds_contained].append(compound_index)

            for pair in pairs:
                i, j = pair
                # msc.2.
                if i > j:
                    continue  # saving time

                contained = False
                for fold in folds:
                    if i in fold[compounds_contained] and j in fold[compounds_contained]:
                        fold[size] += 1
                        contained = True
                        break

                if not contained:
                    omitted_pairs += 1

            # scoring the split found
            loss, variance = split_loss(folds, omitted_pairs)
            print 'omitted_pairs', omitted_pairs
            print 'loss', loss
            if loss < best_loss:
                print '\n'

                best_loss = loss
                best_splits.append(folds)
                if len(best_splits) > n_sets:
                        best_splits = best_splits[-n_sets:]
                losses.append(best_loss)
                variances.append(variance)
                n_omitted_pairs.append(omitted_pairs)
                plt.plot(range(len(losses)), losses, c='r')
                plt.plot(range(len(variances)), variances, c='g')
                plt.plot(range(len(n_omitted_pairs)), n_omitted_pairs, c='b')
                plt.title('best loss over time')
                plt.show()
                print "loss", loss, 'variance', variance, 'omitted_pairs', omitted_pairs
                print 'folds\' sizes', [fold[size] for fold in folds]
                print 'omitted_pairs', omitted_pairs
                print "#BEST SPLIT\n", best_splits[-1]

            print '_________________________________________________\n'
    except KeyboardInterrupt:
        print 'kotek'

    return best_splits


if __name__ == "__main__":
    import sys
    assert len(sys.argv) > 1, 'Please provide `agnieszka` to run agnieszka_splits and `staszek` to run staszek_splits'
    if sys.argv[1] == 'agnieszka':
        print agnieszka_splits('Similarity.csv', 'Random_compounds_100.sdf', 'pairs')
    elif sys.argv[1] == 'staszek':
        print staszek_splits('Similarity.csv', 'Random_compounds_100.sdf', 'pairs')
    else:
        print 'Please provide `agnieszka` to run agnieszka_splits and `staszek` to run staszek_splits'
