#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:14:33 2017

@author: egidiomarotta
"""
import numpy as np
class Adaline(object):
    """ Adaptive Linear Neuron Classifier
    Parameter
    ---------
    eta:float
        Learning rate (0.0 to 1.0)
    n_iter:int
        Passes over the training dataset
        
    Attributes
    ----------
    w_:1d-array
       weights after fitting
    error_:list
    Number of misclassifications in every epoch
    """
    def __init__(self, eta = 0.01, n_iter = 50):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self,X,y):
        """Fitting training data.
        
        Parameter
        ---------
        X: {array-like}, shape = [n_samples, n_features]
            Training vectors, n_sample is the no. of samples and n_features is the no. of features.
        y: array-like, shape = [n_samples]
            Target values
        Returns
        ----------
        self:object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
                output = self.net_input(X)
                errors = (y-output)
                self.w_[1:] += self.eta*X.T.dot(errors)
                self.w_[0] += self.eta*errors.sum()
                cost = (errors**2).sum()/2.0
                self.cost_.append(cost)
        return self
    
    def net_input(self,X):
        """calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self,X):
        return self.net_input(X)
    
    def predict(self,X):
        """returns class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
   
     
    
    
