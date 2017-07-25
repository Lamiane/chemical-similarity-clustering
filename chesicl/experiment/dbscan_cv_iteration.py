#!/usr/bin/env python
# now the line above is super important!

# define single iteration of cross-validation
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import logging
logging.basicConfig()
from drgmum.toolkit import burrito
from chesicl.data_loading.exp_data_loader import load_data
from chesicl.experiment.utils import string_enhancer, serialise_confusion_matrix
from CONFIG import scoring_function

def _dbscan_hyps_valid(hyps):
     # metric 'cosine' not valid for algorithm 'ball_tree'
    # metric 'cosine' not valid for algorithm 'kd_tree'
    # metric 'braycurtis' not valid for algorithm 'kd_tree'
    # metric 'canberra' not valid for algorithm 'kd_tree'
    # metric 'correlation' not valid for algorithm 'ball_tree'
    # metric 'correlation' not valid for algorithm 'kd_tree'
    # metric 'dice' not valid for algorithm 'kd_tree'
    # metric 'hamming' not valid for algorithm 'kd_tree'
    # metric 'jaccard' not valid for algorithm 'kd_tree'
    # metric 'kulsinski' not valid for algorithm 'kd_tree'
    # metric 'matching' not valid for algorithm 'kd_tree
    # metric 'rogerstanimoto' not valid for algorithm 'kd_tree'
    # metric 'russellrao' not valid for algorithm 'kd_tree'
    # metric 'sokalmichener' not valid for algorithm 'kd_tree'
    # metric 'sokalsneath' not valid for algorithm 'kd_tree'
    # metric 'sqeuclidean' not valid for algorithm 'kd_tree'
    # metric 'yule' not valid for algorithm 'kd_tree'
    # metric 'sqeuclidean' not valid for algorithm 'ball_tree'
    if 'cosine' in hyps.values() and 'ball_tree' in hyps.values():
        return False
    if 'cosine' in hyps.values() and 'kd_tree' in hyps.values():
        return False
    if 'braycurtis' in hyps.values() and 'kd_tree' in hyps.values():
        return False
    if 'canberra' in hyps.values() and 'kd_tree' in hyps.values():
        return False
    if 'correlation' in hyps.values() and 'ball_tree' in hyps.values():
        return False
    if 'correlation' in hyps.values() and 'kd_tree' in hyps.values():
        return False
    if 'dice' in hyps.values() and 'kd_tree' in hyps.values():
        return False
    if 'kd_tree' in hyps.values() and hyps['metric'] in ['hamming', 'jaccard', 'kulsinski', 'matching', 'rogerstanimoto']:
        return False
    if 'kd_tree' in hyps.values() and hyps['metric'] in ['russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
        return False
    if 'ball_tree' in hyps.values() and  hyps['metric'] in ['sqeuclidean', 'yule']:
        return False
    return True

def get_hparams():
    # let's not use mahalanobis nor seuclidean at all as it requires tuning
    metrics_space =  ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] + \
    ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming',
       'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto',
       'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    low = -10
    high = 5
    
    # define hiperparameters
    eps = np.logspace(low, high, np.abs(low)+np.abs(high)+1)
    #eps = ["{0:.10f}".format(x) for x in eps]
    min_samples = xrange(2, 11)
    metric = list(metrics_space)
    algorithm = ['ball_tree', 'kd_tree', 'brute']
    leaf_size = [3, 10, 20, 30]
    p = xrange(1, 11)

    dbscan_hiperparameters=[{'n_jobs':2, 'eps':e, 'min_samples':m_s,
                             'metric':mtr, 'algorithm':alg, 'leaf_size':l_s, 'p':_p}
                            for e in eps for m_s in min_samples 
                            for mtr in metric for alg in algorithm for l_s in leaf_size for _p in p]
    
    dbscan_hiperparameters = [x for x in dbscan_hiperparameters if _dbscan_hyps_valid(x)]
    return dbscan_hiperparameters

# default parameters are must-be!
def cv_iteration(n_jobs=2, eps=1., min_samples=30, metric='euclidean', algorithm='brute', leaf_size=30, p=2.):
    X, y_train, _ = load_data()
    scores = []
    model = DBSCAN(n_jobs=n_jobs, eps=eps, min_samples=min_samples, metric=metric, algorithm=algorithm, leaf_size=leaf_size, p=p)
    predictions = model.fit_predict(X)
    scores.append(scoring_function(y_train, predictions))
    return {'result': scores,
            # 'confusion': serialise_confusion_matrix(confusion_matrix(y_train, predictions)),
            'score_name': string_enhancer(str(scoring_function))}

if __name__ == "__main__":
     burrito.print_as_json(burrito.wrap(cv_iteration, save_results=True))
