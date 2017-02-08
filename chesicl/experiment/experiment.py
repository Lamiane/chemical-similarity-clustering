import os
import numpy as np
from drgmum.toolkit.job_pool import JobPool, dict_hash
from drgmum.toolkit import burrito
from scorer import my_binary_score

debug = False

def runner(script_name, script_params, **kwargs):
    # Assumes script is wrapped in src.script_wrapper
    script_params['base_fname'] = os.path.join(kwargs['_job_output_dir'], kwargs['_job_key'])

    cmd = "{} {}".format(script_name, " ".join(
        "--{} {}".format(k, v) for k, v in script_params.iteritems()))
    # Return stderr. Flushing it because it contains logs
    from drgmum.toolkit.utils import exec_command
    _, stderr, ret = exec_command(cmd, flush_stdout=False, flush_stderr=False)
    if ret != 0:
        raise RuntimeError("Failed running cmd")

        

# config
output_data_dir = "../.."
scripts_dir = "/home/pocha/chemical-similarity-clustering/chesicl/experiment"
scoring_function = my_binary_score  # used in each file definig cv_iteration
max_clusters = 100
n_clusters = list(range(1, max_clusters+1))
# let's not use mahalanobis nor seuclidean at all, it requires tuning
metrics_space =  ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] + ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']

if __name__=="__main__":

    # KMEANS
    # define hiperparameters
    init =['k-means++', 'random']
    algorithm = ['full', 'elkan']
    kmeans_hiperparameters = [{'n_clusters':nc, 'init':init_, 'algorithm':alg, 'n_init':1, 'max_iter':300, 'n_jobs':1} 
                       for nc in n_clusters for init_ in init for alg in algorithm]
    if debug:
        kmeans_hiperparameters = kmeans_hiperparameters[:3]
        print 'only three hiperparameters sets'
    # run jobs
    kmeans_job_pool = JobPool(n_jobs=10, output_dir=os.path.join(output_data_dir, 'KMEANS'))
    kmeans_jobs = [{"script_name": os.path.join(scripts_dir, "kmeans_cv_iteration.py"), "script_params": h} for h in kmeans_hiperparameters]
    kmeans_keys = [str(j["script_params"]) for j in kmeans_jobs]
    kmeans_job_pool.map(runner, kmeans_jobs, kmeans_keys)


    # DBSCAN
    # define hiperparameters
    low = -10
    high = 5
    eps = np.logspace(low, high, np.abs(low)+np.abs(high)+1)
    #eps = ["{0:.10f}".format(x) for x in eps]
    min_samples = xrange(2, 11)
    metric = list(metrics_space)
    algorithm = ['ball_tree', 'kd_tree', 'brute']
    leaf_size = [3, 10, 20, 30]
    p = xrange(1, 11)
    n_jobs=4

    dbscan_hiperparameters=[{'n_jobs':n_jobs, 'eps':e, 'min_samples':m_s, 'metric':mtr, 'algorithm':alg, 'leaf_size':l_s, 'p':_p} for e in eps for m_s in min_samples for mtr in metric for alg in algorithm for l_s in leaf_size for _p in p]
    
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
    def _dbscan_hyps_valid(hyps):
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
    
    dbscan_hiperparameters = [x for x in dbscan_hiperparameters if _dbscan_hyps_valid(x)]
    
    if debug:
        dbscan_hiperparameters = dbscan_hiperparameters[:5]
        print 'only three hiperparameters sets'
    # run jobs
    dbscan_job_pool = JobPool(n_jobs=10, output_dir=os.path.join(output_data_dir, 'DBSCAN'))
    dbscan_jobs = [{"script_name": os.path.join(scripts_dir, "dbscan_cv_iteration.py"), "script_params": h} for h in dbscan_hiperparameters]
    dbscan_keys = [str(j["script_params"]) for j in dbscan_jobs]
    dbscan_job_pool.map(runner, dbscan_jobs, dbscan_keys)


