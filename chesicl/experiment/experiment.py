# IMPORTANT! Configuration is in CONFIG.py
import os
import sys
import numpy as np
from drgmum.toolkit.job_pool import JobPool
import CONFIG   # experiment configuration


def runner(script_name, script_params, **kwargs):
    # Assumes script is wrapped in src.script_wrapper
    script_params['base_fname'] = os.path.join(kwargs['_job_output_dir'], kwargs['_job_key'])

    cmd = "{} {}".format(script_name, " ".join(
        "--{} {}".format(k, v) for k, v in script_params.iteritems()))
    # Return stderr. Flushing it because it contains logs
    from drgmum.toolkit.utils import exec_command
    _, stderr, ret = exec_command(cmd, flush_stdout=False, flush_stderr=False)
    if ret != 0:
        raise RuntimeError("Failed running cmd: "+str(cmd))
        

if __name__=="__main__":
    # TODO: prompt przypominajacy o konfuguracji
    print "Have you updated the configuration? [Y]es otherwise no."
    prompt = '> '
    answer = raw_input(prompt)
    if answer != 'Y':
        print 'Update the configuration first!'
        sys.exit(0)
    
    for script_name, hpars, res_dir in zip(CONFIG.scripts_names, CONFIG.hparams, CONFIG.result_folders):
        job_pool = JobPool(n_jobs=CONFIG.n_jobs, output_dir=os.path.join(CONFIG.output_data_dir, res_dir))
        jobs = [{"script_name": os.path.join(CONFIG.scripts_dir, script_name), "script_params": h} for h in hpars]
        keys = [str(j["script_params"]).replace("{",'').replace("}",'') for j in jobs]
        job_pool.map(runner, jobs, keys)



