# config for experiment
from scorer import balanced_accuracy
scoring_function = balanced_accuracy  # used in each file definig cv_iteration
import kmeans_cv_iteration
import dbscan_cv_iteration
import agglomerative_cv_iteration

output_data_dir = "../.."
scripts_dir = "/home/pocha/chemical-similarity-clustering/chesicl/experiment"
max_clusters = 100
n_jobs = 10

# --------- this should be often updated ---------
scripts_names = ["kmeans_cv_iteration.py", "dbscan_cv_iteration.py", "agglomerative_cv_iteration.py"]
hparams = [kmeans_cv_iteration.get_hparams(max_clusters), dbscan_cv_iteration.get_hparams(), agglomerative_cv_iteration.get_hparams(max_clusters)]
result_folders = ['KMEANS_tmpe', 'DBSCAN_tmpe', 'AGGL_tmpe']
# ------------------------------------------------

# # # # temp # # # #
scripts_names = ["agglomerative_cv_iteration.py"]
hparams = [agglomerative_cv_iteration.get_hparams(max_clusters)]
result_folders = ['AGGL_tmpe']
# # # end temp # # # 

assert len(result_folders) == len(scripts_names)
assert len(scripts_names) == len(hparams)
