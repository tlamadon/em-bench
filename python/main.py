# main python file that calls the different implementations
import argparse
import time
import json
import numpy as np
from data_generator_initi_param import Data_Generator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":


    # getting the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--estimator",help="Em estimator object (numpy, pytorch, scikit, tensorflow)")
    parser.add_argument("-t","--tres",type=float,help="Treshold",default=False)
    parser.add_argument("-i","--iter",type=int,help = "number of iterations",default=100)
    parser.add_argument("-n","--nobs",type=int,help = "number of observations",default=10000)
    parser.add_argument("-o","--out",help = "outputfile",default="results.json")
    parser.add_argument("--logfile",help = "log output to a logfile",default="")
    args, unknown = parser.parse_known_args()

    # setting the true parameters
    mean_std, alpha = [[3,1],[4,2]],[0.4,0.6]

    # setting starting values
    alpha__0, mean_mu__0, var_v__0 = np.array([0.5, 0.5]), np.array([[1.],[1.]]), np.array([[[1.]],[[1.]]]) 

    # we start by creating the data
    data= Data_Generator (mean_std, alpha, args.nobs)

    if args.estimator=="numpy":
        from numpy_vect_m import EM_numpy
        estimator = EM_numpy()

    if args.estimator=="pytorch":
        from pytorch_m import EM_pytorch
        estimator = EM_pytorch()

    if args.estimator=="scikit":
        from scikit_m import EM_scikit
        estimator = EM_scikit()

    if args.estimator=="tf1":
        from tensorflow_m import EM_tensorflow
        estimator = EM_tensorflow()

    if args.estimator=="tf2":
        from tf_object_oriented import EM_tensorflow
        estimator = EM_tensorflow(data, alpha__0, mean_mu__0, var_v__0)

    start_ = time.time()
    estimator.fit(data, alpha__0, mean_mu__0, var_v__0, numb_iter= args.iter)
    time_ =time.time() - start_

    # store the results
    dic_result = {'N': args.nobs, 'iter': args.iter, 'time': time_,'estimator': args.estimator}

    print("estimator:{} N:{} iter:{} time:{}".format(args.estimator,args.nobs,args.iter,time_))

    with open(args.out, 'w') as fp:
        json.dump(dic_result, fp) 


