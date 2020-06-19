DOIT_CONFIG = {'check_file_uptodate': 'timestamp'}

def task_all():
    """Simulate bunch of results"""

    for method in ['numpy','pytorch','scikit','tf2']:
        for nobs in [10000,100000,1000000,10000000]:
            for niter in [100,1000]:
                for rep in range(10):
                    name = "py_E{}_N{}_I{}_R{}".format(method, nobs, niter, rep)
                    filename = "sims/res_E{}_N{}_I{}_R{}.json".format(method, nobs, niter, rep)

                    yield {
                        'name': name,
                        'actions': ["python python/main.py --iter {} --nobs {} --estimator {} -o {}".format(niter, nobs, method, filename)],
                        'targets': [filename],
                    }

        
