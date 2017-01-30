from sklearn.model_selection import ParameterGrid
import numpy as np
from load_constraints import load_similarity_matrices
from cPickle import dump


def score_function_binary(clusters, constraints):
    return 1, 2, 3, 4


def score_function_scale(clusters, constraints):
    return 1, 2, 3, 4


# uwaga: nie martwimy sie tym jak poprawnie dobierac liczbe klastrow, na razie sprawdzamy wszystkie


def run(model_class, param_grid, fingerprints, constraints):
    hyperparams = list(ParameterGrid(param_grid))
    results = {}

    for hparams in hyperparams:
        my_model = model_class(hparams)
        mini_1 = []
        mini_2 = []
        mini_3 = []
        scale = []
        for few in xrange(5):
            my_model.fit(fingerprints)
            clusters = my_model.predict(fingerprints)
            a, b, c, d = score_function_binary(clusters, constraints)
            mini_1.append(a)
            mini_2.append(b)
            mini_3.append(c)
            scale.append(score_function_scale(clusters, constraints))
        results[hparams] = (mini_1, mini_2, mini_3, scale)
    return results

# mozliwe ulepszenia: chcielibysmy badac GDZIE model sie myli


# TODO: zbadac ten model czy dziala jak ja chciala
class RandomModel(object):
    def __init__(self, k, seed=None):
        self.k = k
        if seed is None:
            seed = np.random.rand()
        self.seed = seed

    def fit(self, X):
        # do nothing
        pass

    def predict(self, X):
        assert isinstance(data, np.ndarray)
        np.random.seed(self.seed)
        return np.random.randint(self.k, data.shape[0])


if __name__ == "__main__":
    models = []
    hpgrids = []
    data = []  # load data
    constraints = load_similarity_matrices(folder_with_pairs='data/pairs',
                                           all_compounds_file='data/Random_compounds_100.sdf',
                                           similarity_matrix_file='data/Similarity.csv')
    for pair in zip(models, hpgrids):
        model, grid = pair[0], pair[1]
        results = run(model, grid, data, constraints)
        with open(str(model)+'.pkl') as f:
            dump(results, f)
