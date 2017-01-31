import os
import numpy as np
from drgmum.toolkit.job_pool import JobPool, dict_hash
from drgmum.toolkit import burrito
from scorer import my_binary_score

def runner(script_name, script_params, **kwargs):
    # Assumes script is wrapped in src.script_wrapper
    script_params['base_fname'] = os.path.join(kwargs['_job_output_dir'], kwargs['_job_key'])

    cmd = "{} {}".format(script_name, " ".join(
        "--{} {}".format(k, v) for k, v in script_params.iteritems()))

    # Return stderr. Flushing it because it contains logs
    from drgmum.toolkit.utils import exec_command
    _, stderr, ret = exec_command(cmd, flush_stdout=False, flush_stderr=True)
    if ret != 0:
        raise RuntimeError("Failed running cmd")

# config
output_data_dir = "../.."
scripts_dir = "/home/pocha/chemical-similarity-clustering/chesicl/experiment"
scoring_function = my_binary_score  # used in each file definig cv_iteration
max_clusters = 200
n_clusters = list(range(1, max_clusters))
metrics_space =  ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] + ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
     
    
# KMEANS
# define hiperparameters
init =['k-means++', 'random']
algorithm = ['full', 'elkan']
kmeans_hiperparameters = [{'n_clusters':nc, 'init':init_, 'algorithm':alg, 'n_init':1, 'max_iter':600} 
                   for nc in n_clusters for init_ in init for alg in algorithm]
# run jobs
kmeans_job_pool = JobPool(n_jobs=10, output_dir=os.path.join(output_data_dir, 'KMEANS'))
kmeans_jobs = [{"script_name": os.path.join(scripts_dir, "kmeans_cv_iteration.py"), "script_params": h} for h in kmeans_hiperparameters]
kmeans_keys = [str(j["script_params"]) for j in kmeans_jobs]
kmeans_job_pool.map(runner, kmeans_jobs, kmeans_keys)

del kmeans_hiperparameters, kmeans_jobs, kmeans_keys, init, algorithm
        

# DBSCAN
# define hiperparameters
low = -10
high = 5
eps = np.logspace(low, high, np.abs(low)+np.abs(high)+1)
min_samples = xrange(2, 11)
metric = list(metrics_space)
algorithm = ['ball_tree', 'kd_tree', 'brute']
leaf_size = [3, 10, 20, 30]
p = xrange(1, 11)
n_jobs=4

dbscan_hiperparameters=[{'n_jobs':n_jobs, 'eps':e, 'min_samples':m_s, 'metric':mtr, 'algorithm':alg, 'leaf_size':l_s, 'p':_p} for e in eps for m_s in min_samples for mtr in metric for alg in algorithm for l_s in leaf_size for _p in p]

# run jobs
dbscan_job_pool = JobPool(n_jobs=10, output_dir=os.path.join(output_data_dir, 'KMEANS'))
dbscan_jobs = [{"script_name": os.path.join(scripts_dir, "dbscan_cv_iteration.py"), "script_params": h} for h in dbscan_hiperparameters]
dbscan_keys = [str(j["script_params"]) for j in dbscan_jobs]
dbscan_job_pool.map(runner, dbscan_jobs, dbscan_keys)

del low, high, eps, min_samples, metric, algorithm, leaf_size, p, n_jobs, dbscan_hiperparameters, dbscan_keys

