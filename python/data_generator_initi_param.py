import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from scipy.stats import norm 
import time 
import json
import torch 

def Data_Generator(mean_std, alpha, num_points):


    
    
    """
        Generate the Data for every method 
        :param means_std: Array of the mean and Std of the data to generate : np.array ([mean_1,std_1],[mean_2,std_2 ] ... )  
        :param alpha: Array of weights of the data to generate 
        :param num_points: Number of points of the data to generate
        :return: Data Generated ( Numpy Shape )
    """
        
    random_index  = np.random.choice(len(alpha), size=num_points, replace=True, p=alpha)  

    data_generated = np.array([norm.rvs(*mean_std[index]) for index in random_index])
    
    return (data_generated)

def Initial_Parameters(k):
    
    """
        Generate randomly the initial parameter for every method
        :param k: Number of Clusters
        :return: weights,mean,std_deviation 
    """
    
    # Numpy and Tensorflow shape
    
    alpha_numpy_tf = np.random.random(k) 
    alpha_numpy_tf /= alpha_numpy_tf.sum()  # Constraint sum(weights)=1 
    mean_mu_numpy_tf = np.random.random((k,1))  
    std_sig_numpy_tf = np.array([np.eye(1)]* k)

    return (alpha_numpy_tf,mean_mu_numpy_tf,std_sig_numpy_tf)


