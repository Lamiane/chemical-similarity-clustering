import os
from drgmum.toolkit.job_pool import JobPool, dict_hash
from drgmum.toolkit import burrito
from kmeans_cv_iteration import hiperparameters as kmeans_hiperparameters

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

# KMEANS
job_pool = JobPool(n_jobs=10, output_dir='KMEANS')
jobs = [{"script_name": "/home/pocha/chemical-similarity-clustering/kmeans_cv_iteration.py", "script_params": h} for h in kmeans_hiperparameters]
keys = [str(j["script_params"]) for j in jobs]
job_pool.map(runner, jobs, keys)
