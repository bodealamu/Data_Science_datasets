#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:34:33 2017

@author: egidiomarotta
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
from Perceptron_UnitStep import Perceptron
## Use same same procedure to import ADALINE and work on it as perceptron (you will be doing the same steps)

## read csv file Iris_DataSet.csv name it df = ??
print(df.tail())

y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa', -1,1)
X = df.iloc[0:100, [0,2]].values

print(y)
print(X)
plt.scatter(X[0:50,0], X[0:50,1], color ='r', marker = 'o', s = 10, label = 'setosa')
plt.scatter(X[50:100,0], X[50:100,1], color ='b', marker = 'x', s = 10, label = 'versicolor')
plt.ylim(0,6)
plt.xlim(4,7.5)
plt.xlabel('petal length (cm)')
plt.ylabel('sepal length (cm)')
plt.title('Setosa, Versicolor Length Dimensions', fontsize=12)
plt.legend(loc = 'upper left')
pylab.savefig('perception.png',dpi=300)
plt.show()

plt.close('all')

#Spliting the data set into training and test sets
from sklearn.cross_validation import train_test_split
## Here train your data "X_train, X_test, y_train, y_test = ??"

#Scaling the X matrix
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
## here you transform X_train_std = ??
## here you transform X_test_std = ??



## use perceptron function ppn = Perceptron( ? , ? )
## fit your trained data X_train and y_train

#Prediction of class labels based on the trained data and the generated weights
## predict function y_pred = ppn.predict(?)

print('Misclassified samples: %d' % (y_test != y_pred).sum())

#Calculation of the Accuracy of the Classification
from sklearn.metrics import accuracy_score
## use accuracy_score function on y_test and y_pred and print results

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, color = 'r', marker ='o', label ='Iterations')
plt.xlabel('Epoch')
plt.ylabel('# of Misclassifications')
plt.legend(loc = 'upper right')
pylab.savefig('perception_epoches.png',dpi=300)
plt.show()

plt.close('all')
from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y, classifier, resolution = 0.02):
    # setup the markers nd colors map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    print('------------------------------------------')

#plot the decision space
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


    #plot class sample
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y ==cl, 1], s= 10, alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)


plt.close('all')
plot_decision_regions(X,y, classifier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.title('Decision Surface', fontsize=12)
pylab.savefig('decision_surface.png',dpi=300)
plt.show()



