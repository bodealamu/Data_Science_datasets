
# coding: utf-8

# In[358]:

"""
Alamu Olabode Afolabi
1498663
Guide to Engineering Data Science
Homework 7

Nov 16 2017

"""


# In[359]:

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[360]:

# import the dataframes
df1 = pd.read_excel('uppermoon1.xls')
df2 = pd.read_excel('uppermoon2.xls')
df3 = pd.read_excel('lowermoon1.xls')
df4 = pd.read_excel('lowermoon2.xls')


# In[361]:

from sklearn.metrics import classification_report, confusion_matrix


# In[362]:

print(df1.head())


# In[363]:

print(df2.head())


# In[364]:

print(df3.head())


# In[365]:

print(df4.head())


# In[366]:

# Create a new column in the lowermoo1 dataset called class
df3['Class'] = 1
df1['Class'] = 2
# Create a new column in the lowermoo1 dataset called class
df4['Class'] = 1
df2['Class'] = 2


# In[367]:

print(df3.head())


# In[368]:

print(df1.head())


# In[369]:

# Append both dataframes together
linear = df3.append(df1,ignore_index = True)
Non_linear = df4.append(df2,ignore_index = True)


# In[370]:

print(linear.tail())


# In[371]:

print(linear.head())


# In[372]:

print(Non_linear.tail())


# In[373]:

print(Non_linear.head())


# In[374]:

# Visualise the data
plt.scatter(Non_linear['x'], Non_linear['y'],c = Non_linear['Class'] )
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Plot of Non linear separable points.')
plt.show()


# In[375]:

plt.scatter(linear['x'], linear['y'],c = linear['Class'] )
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Plot linearly separable points.')
plt.show()


# In[376]:

# Check for missing values in both datasets
print(linear.isnull().sum())


# In[377]:

# for nonlinear dataset
print(Non_linear.isnull().sum())


# In[378]:

print('No missing values in either datasets')


# In[437]:

# Standaradise both datasets
# Import the library
from sklearn.preprocessing import StandardScaler
print('--------------------------------------------')
print('STandardize dataset')


# In[438]:

# for the linear dataset
scaled = StandardScaler()
scaled.fit(linear)
linear = scaled.transform(linear)


# In[439]:

# for non linear dataset
scaled.fit(Non_linear)
Non_linear = scaled.transform(Non_linear)


# SPlit the data into training and test

# In[441]:

# Split the data into training and data
from sklearn.cross_validation import train_test_split
print('-----------------------------------------------')
print('Train, test, split')


# In[442]:

# get the X matrix and y matrix for each dataset
# for linear
X_linear = linear[:,[0,1]]
Y_linear = linear[:,2]
# for non linear separable dataset
X_nonlinear = Non_linear[:,[0,1]]
Y_nonlinear = Non_linear[:,2]


# In[ ]:




# Linear dataset classification

# In[443]:

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_linear, Y_linear, test_size=0.3)


# Decision tree for linearly separable dataset

# In[444]:

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train_l,y_train_l)
linear_prediction = dt.predict(X_test_l)


# In[445]:

print('Decision tree for linear separable')
print(confusion_matrix(y_test_l,linear_prediction))


# In[446]:

print(classification_report(y_test_l,linear_prediction))


# Random Forest for linear dataset

# In[447]:

# Import random forest classifier
from sklearn.ensemble import RandomForestClassifier
# instantiate it
rfc = RandomForestClassifier()
# fit the data
rfc.fit(X_train_l,y_train_l)


# In[449]:

# make predictions off the data
rfc_linear_prediction = rfc.predict(X_test_l)
print('Random forest result linear dataset')


# In[450]:

print(confusion_matrix(y_test_l,rfc_linear_prediction))


# In[451]:

print(classification_report(y_test_l,rfc_linear_prediction))


# Support Vector Machine Classification for linearly separable dataset

# In[452]:

from sklearn.svm import SVC


# In[453]:

# instantiate
svc = SVC()
# fit the SVM classifier to the training dataset
svc.fit(X_train_l,y_train_l)


# In[456]:

# Create predictions
svc_linear_prediction = svc.predict(X_test_l)
print('SVM for linear dataset')


# In[457]:

# print confusion matrix and classification report
print(confusion_matrix(y_test_l,svc_linear_prediction))
print(classification_report(y_test_l,svc_linear_prediction))


# In[458]:

print(classification_report(y_test_l,svc_linear_prediction))


# Multilayer perceptron for linearly separable dataset

# In[459]:

from sklearn.neural_network import MLPClassifier


# In[460]:

mlp = MLPClassifier(alpha = 1)
mlp.fit(X_train_l, y_train_l)


# In[462]:

mlp_linear_prediction = mlp.predict(X_test_l)
print('Result for Multilayer perceptron linear dataset')


# In[463]:

# print confusion matrix and classification report
print(confusion_matrix(y_test_l,mlp_linear_prediction))
print(classification_report(y_test_l,mlp_linear_prediction))


# Naive Bayes for linear separable datasets

# In[464]:

from sklearn.naive_bayes import GaussianNB


# In[466]:

# fit and predict
gb = GaussianNB()
gb.fit(X_train_l, y_train_l)
gb_linear_prediction = gb.predict(X_test_l)
print('Naive bayes result linear dataset')


# In[467]:

# print confusion matrix and classification report
print(confusion_matrix(y_test_l,gb_linear_prediction))


# In[468]:

print(classification_report(y_test_l,gb_linear_prediction))


# Perceptron for linearly separable dataset

# In[469]:

from sklearn.linear_model import perceptron
pcp = perceptron.Perceptron(max_iter=100, verbose=0, random_state=None, fit_intercept=True, tol=0.002)
pcp.fit(X_train_l, y_train_l)
pcp_linear_prediction = pcp.predict(X_test_l)
print('Result for perceptron linear dataset')
# print confusion matrix and classification report
print(confusion_matrix(y_test_l,pcp_linear_prediction))
print(classification_report(y_test_l,pcp_linear_prediction))


# In[ ]:




# Non linear dataset classification

# Decision tree for non linear separable dataset

# In[470]:

# SPlit the nonlinearly separable dataset
X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(X_nonlinear, Y_nonlinear, 
                                                                test_size=0.3)


# In[471]:

from sklearn.tree import DecisionTreeClassifier


# In[472]:

dt = DecisionTreeClassifier()


# In[473]:

dt.fit(X_train_nl,y_train_nl)


# In[475]:

nonlinear_prediction = dt.predict(X_test_nl)
print('Decsision tree for Nonlinear result')


# In[476]:

# print the confusiion matrix
print(confusion_matrix(y_test_nl,nonlinear_prediction))


# In[477]:

# print classification report
print(classification_report(y_test_nl,nonlinear_prediction))


# Random forest for non linear dataset

# In[478]:

# Import random forest classifier
from sklearn.ensemble import RandomForestClassifier
# instantiate it
rfc = RandomForestClassifier()
# fit the data
rfc.fit(X_train_nl,y_train_nl)


# In[480]:

# make predictions off the data
rfc_nonlinear_prediction = rfc.predict(X_test_nl)
print('Random forest result for nonlinear dataset')


# In[481]:

print(confusion_matrix(y_test_nl,rfc_nonlinear_prediction))


# In[482]:

print(classification_report(y_test_nl,rfc_nonlinear_prediction))


# Support Vector Machines for non linear separable dataset

# In[483]:

# import the classifier
from sklearn.svm import SVC


# In[484]:

# Instantiate it
svc = SVC()
# fit to the training data
svc.fit(X_train_nl, y_train_nl)


# In[486]:

# make predictions
svc_nonlinear_prediction = svc.predict(X_test_nl)
print('SVM result for non linear dataset')


# In[487]:

# print classification report and confusion matrix
print(confusion_matrix(y_test_nl,svc_nonlinear_prediction))


# In[488]:

print(classification_report(y_test_nl,svc_nonlinear_prediction))


# Multilayer perceptron for nonlinearly separable dataset

# In[490]:

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(alpha = 1)
mlp.fit(X_train_nl, y_train_nl)
mlp_nonlinear_prediction = mlp.predict(X_test_nl)
print('MLP classifier for non linear dataset result')
# print confusion matrix and classification report
print(confusion_matrix(y_test_nl,mlp_nonlinear_prediction))
print(classification_report(y_test_nl,mlp_nonlinear_prediction))


# Perceptron for nonlinear dataset

# In[491]:

from sklearn.linear_model import perceptron


# In[492]:

pcp = perceptron.Perceptron(max_iter=100, verbose=0, random_state=None, fit_intercept=True, tol=0.002)


# In[493]:

pcp.fit(X_train_nl, y_train_nl)


# In[495]:

pcp_nonlinear_prediction = pcp.predict(X_test_nl)
print('Perceptron for non linear dataset')


# In[496]:

# print confusion matrix and classification report
print(confusion_matrix(y_test_nl,pcp_nonlinear_prediction))
print(classification_report(y_test_nl,pcp_nonlinear_prediction))


# In[ ]:




# Naive Bayes for nonlinear separable dataset

# In[497]:

from sklearn.naive_bayes import GaussianNB


# In[498]:

gb = GaussianNB()


# In[499]:

gb.fit(X_train_nl, y_train_nl)


# In[501]:

gb_nonlinear_prediction = gb.predict(X_test_nl)
print('Naive Bayes result for non linear dataset')


# In[502]:

print(classification_report(y_test_nl,gb_nonlinear_prediction))
print('_______________________________________________________')


# Conclusion

# In[507]:

print('Ranking for linearly separable dataset')
print('Decision tree,')
print('Random forest')
print('SVM')
print('MLP')
print('Perceptron')
print('Naive bayes')
print('____________________________________')


# In[509]:

print('Ranking for nonlinearly separable dataset')
print('Random forest')
print('SVM')
print('Decision tree,')


print('MLP')
print('Perceptron')
print('Naive bayes')
print('_________________________________________________________')


# In[510]:

print('The end')


# In[ ]:



