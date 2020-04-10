#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:14:33 2017

@author: egidiomarotta
"""
import numpy as np
class Perceptron(object):
    """ Perceptron Classifier.
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
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self,X,y):
        """Fitting data
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
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update !=0.0)
            self.errors_.append(errors)
            print(self.w_)
        return self
    
    def net_input(self,X):
        """calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self,X):
        """returns class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
       
    
    
