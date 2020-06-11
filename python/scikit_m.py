import numpy as np
from scipy.stats import norm 
import numpy.random as random
from sklearn.mixture import GaussianMixture


class EM_scikit:
    """  Scikit implementation """

    def __init__(self):
        pass
    
    def fit(self,data,alpha,mean_mu,std_sig, treshold=False ,epsilon=1e-5,numb_iter=100): 
        
        
        
        """
            Runs the Expectation-Maximization algorithm using Scikit
            :param data: Data used in the EM algorithm
            :param alpha: Initial guess for weight values
            :param mean_mu: Initial guess for mean values
            :param std_sig: Initial guess for standard deviation values
            :param epsilon: Precision on the likelihood 
            :param numb_iter: Number of iterations 
            :return: Estimated parameters ( weight, mean, standard deviation )
            
        """
        # Reshape
            
        mean_mu=mean_mu.reshape(len(alpha),)
        std_sig=std_sig.reshape(len(alpha),)
        
        
        # Initialization 
        
        data_dimension=data.shape[0]
        nb_mixture_component=alpha.shape[0]
        
        #Create the object Gaussian mixture model  
        
        gaussian_mixture_scikit = GaussianMixture(n_components = nb_mixture_component, tol=epsilon,
                                            covariance_type='spherical',      
                                            max_iter=numb_iter,
                                            means_init=mean_mu.reshape(nb_mixture_component,1),
                                            weights_init=alpha,
                                            precisions_init=std_sig**-2)
        
        # Fit on train Data
        
        gaussian_mixture_scikit.fit(data.reshape(data_dimension,1))

        return gaussian_mixture_scikit.weights_,gaussian_mixture_scikit.means_,np.sqrt(gaussian_mixture_scikit.covariances_)









