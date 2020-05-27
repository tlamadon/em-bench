import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from scipy.stats import norm 
import time 
import json
import torch 

class EM_tensorflow:

    def __init__(self, data, mean_mu__0,var_v__0,alpha__0,
                 k=2,treshold=False,nb_iter=100,eps = 10e-5):
        
        self.data=data
        self.data=self.data.reshape(self.data.shape[0],1)
        self.treshold=treshold 
        self.eps=eps
        self.k=k
        
        var_v__0=var_v__0.reshape(2,1)
        const_gauss = tf.constant(np.log(2 * np.pi) * 1 , dtype=tf.float64)


        # BUILD COMPUTATIONAL GRAPH

        # input of the model 
        
        self.input = tf.placeholder(tf.float64, [None, 1])

        # computing input statistics
        
        mean_0 = tf.reduce_mean(self.input, 0)
        diff_0 = tf.squared_difference(self.input, tf.expand_dims(mean_0, 0))
        var_0 = tf.reduce_sum(diff_0, 0) / tf.cast(tf.shape(self.input)[0], tf.float64)
        var_mean_0 = tf.cast(tf.reduce_sum(var_0) / k / 1 , tf.float64)

        # Initial values for the algorithm 
        
        self.mean_init = tf.placeholder_with_default( mean_mu__0,shape=[k, 1] )
        
        self.var_init = tf.placeholder_with_default( var_v__0,shape=[k, 1] )
        
        self.weight_init = tf.placeholder_with_default(alpha__0 , shape=[k] )
        

        # Variables for training 
        
        self.mean_mu = tf.Variable(self.mean_init, dtype=tf.float64)
        self.var_v = tf.Variable(self.var_init, dtype=tf.float64)
        self.weight_pi = tf.Variable(self.weight_init, dtype=tf.float64)

        # Step 1 : Expectation  ( logsumexp , normal )
        
        diff_squared = tf.squared_difference(tf.expand_dims(self.input, 0), tf.expand_dims(self.mean_mu, 1))
        diff_squared_var = tf.reduce_sum(diff_squared / tf.expand_dims(self.var_v, 1), 2)
        gauss_log = tf.expand_dims(const_gauss + tf.reduce_sum(tf.log(self.var_v), 1), 1)
        gauss_log_com = -1/2 * (gauss_log + diff_squared_var)
        weight_log = gauss_log_com + tf.expand_dims(tf.log(self.weight_pi), 1)
        logg = tf.expand_dims(tf.reduce_max(weight_log, 0), 0)
        exp_logg = tf.exp(weight_log - logg)
        exp_logg_sum = tf.reduce_sum(exp_logg, 0)
        gamma = exp_logg / exp_logg_sum

        # Step 2 : Maximization 
        
        gamma_sum = tf.reduce_sum(gamma, 1)
        gamma_weighted = gamma / tf.expand_dims(gamma_sum, 1)
        mean_estim = tf.reduce_sum(tf.expand_dims(self.input, 0) * tf.expand_dims(gamma_weighted, 2), 1)
        diff_estim = tf.squared_difference(tf.expand_dims(self.input, 0), tf.expand_dims(mean_estim, 1))
        var_estim = tf.reduce_sum(diff_estim * tf.expand_dims(gamma_weighted, 2), 1)
        weights_estim = gamma_sum / tf.cast(tf.shape(self.input)[0], tf.float64)
        var_estim *= tf.expand_dims(gamma_sum, 1)
        var_estim /= tf.expand_dims(gamma_sum, 1)
        
            
        # Compute Loglikelihood 
        
        self.log_likelihood = tf.reduce_sum(tf.log(exp_logg_sum)) + tf.reduce_sum(logg)
        self.mean_log_likelihood = self.log_likelihood / tf.cast(tf.shape(self.input)[0] * tf.shape(self.input)[1], tf.float64)

        # assignement of new values for parameters 
        
        self.train_step = tf.group( self.mean_mu.assign(mean_estim), 
                                   self.var_v.assign(var_estim), self.weight_pi.assign(weights_estim) )        
       
 
    def fit (self,nb_iter):
        
        # RUN COMPUTATIONAL GRAPH

        with tf.Session() as sess:
            
            # initializing trainable variables ( cout ++ ( creacte object ) )
            
            sess.run( tf.global_variables_initializer(), feed_dict={self.input: self.data, self.mean_init: self.data[:self.k],} )
          
            log_likelihood_init = -np.inf

            for step in range(nb_iter):
                
                # Training step execution 
                
                log_likelihood, _ = sess.run( [self.mean_log_likelihood, self.train_step], feed_dict={self.input: self.data} )
            

                
                if self.treshold==True: 
                    
                    if step > 0:

                        log_likelihood_diff = log_likelihood - log_likelihood_init
                        

                        if log_likelihood_diff <= self.eps:

                            break


                    log_likelihood_init = log_likelihood


            # Get the final Values 
            
            means_em = self.mean_mu.eval(sess)
            variance_em = self.var_v.eval(sess)
            weight_em=self.weight_pi.eval(sess)
        
            
        return means_em,np.sqrt(variance_em),weight_em