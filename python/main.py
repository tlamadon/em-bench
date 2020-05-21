# main python file that calls the different implementations

import argparse
import time
import json

from Main_Numpy_vect import Expectation_Maximization_Numpy_Vect 
from Main_Pytorch import Expectation_Maximization_Pytorch
from DataGenerator_Initial_Param import Data_Generator

if __name__ == "__main__":

    # getting the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator")
    parser.add_argument("-i","--iter",type=int,help = "number of iterations",default=100)
    parser.add_argument("-n","--nobs",type=int,help = "number of observations",default=10000)
    parser.add_argument("-o","--out",help = "outputfile",default="results.json")
    parser.add_argument("--logfile",help = "log output to a logfile",default="")
    args, unknown = parser.parse_known_args()

    # setting the true parameters
    mean_std, alpha = [[3,1],[4,2]],[0.4,0.6]

    # setting starting values
    alpha__0, mean_mu__0, var_v__0 = 0, 0, 0

    # we start by creating the data
    data= Data_Generator (mean_std, alpha, args.nobs)

    if args.estimator=="numpy":
        start_time = time.perf_counter()
        Expectation_Maximization_Numpy_Vect(data, alpha__0, mean_mu__0, var_v__0, numb_iter= args.iter)
        time_ = time.perf_counter() - start_time

    if args.estimator=="pytorch":
        start_time = time.perf_counter()
        Expectation_Maximization_Pytorch(data, alpha__0, mean_mu__0, var_v__0, numb_iter=args.iter, epsilon=1e-5)
        time_ = time.perf_counter() - start_time

    # store the results
    dic_result = {'N': args.nobs, 'iter': args.iter, 'time': time_,'estimator': args.estimator}

    with open(args.out, 'w') as fp:
        json.dump(dic_result, fp) 

