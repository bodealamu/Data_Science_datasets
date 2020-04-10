
# coding: utf-8

# In[2]:

# Import all libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
get_ipython().magic('matplotlib inline')


# In[4]:

# Import the datasets
Irisdf = pd.read_csv('Iris_DataSet.csv', header = None, names = ['SepalLengthCm','SepalWidthCm',
                                                                 'PetalLengthCm','PetalWidthCm',
                                                                 'Species'])
ElectricPowerdf = pd.read_csv('ElectricPowerData.csv')


# In[5]:

print(Irisdf.head())


# In[6]:

print(Irisdf.isnull().sum()) # print out the number of null values in each column


# In[7]:

# SUmmarizes the key statistics parameters for each column in the dataset
print(Irisdf.describe())


# In[8]:

# Number of unique types of iris flower present
print(Irisdf['Species'].unique())


# In[9]:

print('Number of unique iris flower types present is ', len(Irisdf['Species'].unique()))


# In[10]:

# Data visualizations
sns.pairplot(Irisdf, hue = 'Species')


# In[11]:

# Basic Statististics for Electric power dataset
print(ElectricPowerdf.describe())


# In[12]:

# Checking for missing values
print('Number of missing values per column')
print(ElectricPowerdf.isnull().sum())


# In[13]:

print('Both datasets dont have missing values, hence no need for filling with median')


# In[14]:

Irisdf.columns


# In[15]:

# Developing the matrix of features
X_matrix = Irisdf[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]


# In[16]:

print(X_matrix)


# In[17]:

# Scale the variables in the iris dataset
scaled = StandardScaler() # Instantiate
scaled.fit(X_matrix)
X_matrix_scaled = scaled.transform(X_matrix)


# In[18]:

print(X_matrix_scaled)


# In[19]:

# Separate the electric dataset into y and x values
print(ElectricPowerdf.columns)


# In[20]:

# produce some visualizations on the Electric power dataset
sns.pairplot(ElectricPowerdf)


# In[21]:

# Scatter plot
plt.scatter(y = ElectricPowerdf['Avg Ambient Temp(oF)'], x = ElectricPowerdf['Electric Power'], c = 'r')
plt.scatter(y = ElectricPowerdf['No. Days/Month'], x = ElectricPowerdf['Electric Power'], c = 'g')
plt.scatter(y = ElectricPowerdf['Avg Product Purity(%)'], x = ElectricPowerdf['Electric Power'])
plt.scatter(y = ElectricPowerdf['Product Produced(Tons)'], x = ElectricPowerdf['Electric Power'], c = 'y')
plt.xlabel('Electric power')
plt.grid()
plt.legend()
plt.show()



# In[22]:

#Extract the output variable
Y_ElectricPowerdf= ElectricPowerdf['Electric Power']


# In[23]:

# Extract the features
X_ElectricPowerdf = ElectricPowerdf[['Avg Ambient Temp(oF)', 'No. Days/Month',
       'Avg Product Purity(%)', 'Product Produced(Tons)']]


# In[24]:

# The electric power dataset needs to be split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X_ElectricPowerdf, Y_ElectricPowerdf, 
                                                    test_size = 0.3)


# In[25]:

print(X_train)


# In[26]:

print(X_test)


# In[27]:

print(y_train)


# In[28]:

print(y_test)


# In[29]:

# Linear regression of the Electric power dataset
lm = LinearRegression() # Instantiate 
lm.fit(X = X_train, y = y_train)


# In[30]:

# Show the regression coefficients
lm.coef_


# In[31]:

lm.intercept_


# In[32]:

# Produce a dataframe of the coefficients
cdf = pd.DataFrame(lm.coef_, X_ElectricPowerdf.columns , columns = ['Coefficient'])


# In[33]:

print(cdf)


# In[34]:

# Time to make predictions
predictions = lm.predict(X_test)


# In[35]:

print(predictions)


# In[36]:

print(y_test)


# In[37]:

plt.scatter(y_test, predictions)
plt.xlabel('y test values')
plt.ylabel('predictions')
plt.show()


# Logistic regression of Iris dataset

# In[38]:

# Extract first 100 rows of the scaled dataset for X and Y
X_iris_scaled = X_matrix_scaled[0:100,:]


# In[39]:

print(X_iris_scaled.shape)


# In[40]:

#Irisdf['Species'].iloc[range(100)]


# In[41]:

#Irisdf['Specie Code'] = 1


# In[42]:

Irisdf_100 = Irisdf[0:100] # Extracts the first 100 rows of the dataset


# In[43]:

Irisdf_100['Specie Code'] = np.nan


# In[44]:

#LOG = Irisdf_100['Species'] == 'Iris-setosa'


# In[45]:

#Irisdf_100[LOG].fillna(1)


# In[46]:

print(Irisdf_100)


# In[ ]:




# In[47]:

# Create a target column with 0 representing Iris-setosa and 1 Iris-versicolor


# In[48]:

Irisdf_100.fillna(value = 0, limit = 50, inplace = True)


# In[49]:

Irisdf_100.fillna(1, inplace = True)


# In[50]:

print(Irisdf_100)


# In[51]:

print(X_iris_scaled)


# In[52]:

Y_values_iris = Irisdf_100['Specie Code']


# In[53]:


plt.scatter(X_iris_scaled[:,3:], Y_values_iris, c = 'r')
plt.xlabel('Petal length')
plt.ylabel('Classification')
plt.show()


# In[ ]:




# In[54]:

# Split the data into training and testing
X_train_log , X_test_log , y_train_log, y_test_log = train_test_split(X_iris_scaled,Y_values_iris, test_size = 0.3)


# In[55]:

# Fit the data using logistic regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train_log,y_train_log)


# In[56]:

predictions = logmodel.predict(X_test_log)


# In[57]:

print(predictions)


# In[58]:

from sklearn.metrics import classification_report


# In[59]:

print(classification_report(y_test_log,predictions))


# Perceptron

# In[60]:

#!/usr/bin/env python3

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
        


# In[64]:

#PERCEPTRON
from sklearn.preprocessing import StandardScaler




# In[66]:

#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


# In[67]:

from sklearn.cross_validation import train_test_split


# In[69]:

ppn = Perceptron(eta = 0.1, n_iter = 10)


# In[70]:


ppn.fit(X_train_log, y_train_log)
#Prediction of class labels based on the trained data and the generated weights
y_pred = ppn.predict(X_test_log)
print('Misclassified samples: %d' % (y_test_log != y_pred).sum())


# In[72]:


#Calculation of the Accuracy of the Classification

## use accuracy_score function on y_test and y_pred and print results
print('Accuracy: %.2f' % accuracy_score(y_test_log, y_pred))


# In[76]:

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, color = 'r', marker ='o', label ='Iterations')
plt.xlabel('Epoch')
plt.ylabel('# of Misclassifications')
plt.legend(loc = 'upper right')
#pylab.savefig('perception_epoches.png',dpi=300)
plt.show()

plt.close('all')


# In[88]:

def plot_decision_regions(X_iris_scaled,Y_values_iris, classifier, resolution = 0.02):
    # setup the markers nd colors map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(Y_values_iris))])
    print('------------------------------------------')

#plot the decision space
    x1_min, x1_max = X_iris_scaled[:, 0].min()-1, X_iris_scaled[:, 0].max()+1
    x2_min, x2_max = X_iris_scaled[:, 1].min()-1, X_iris_scaled[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class sample
    for idx, cl in enumerate(np.unique(Y_values_iris)):
        plt.scatter(x=X_iris_scaled[y == cl, 0], y=X_iris_scaled[y ==cl, 1], s= 10, alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)

plt.close('all')


# In[90]:



plot_decision_regions(X_iris_scaled,Y_values_iris, classifier = ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.title('Decision Surface', fontsize=12)
#pylab.savefig('decision_surface.png',dpi=300)
plt.show()



# In[ ]:




# 

# Alpine

# In[91]:

#!/usr/bin/env python3

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


# In[92]:

from ADALINE import Adaline
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model


# In[96]:

ada = Adaline(eta = 0.01, n_iter = 50)
ada.fit(X_train_log, y_train_log)
#Prediction of class labels based on the trained data and the generated weights
y_pred = ada.predict(X_test_log)
print('Misclassified samples: %d' % (y_test_log != y_pred).sum())


# In[100]:

#Calculation of the Accuracy of the Classification
from sklearn.metrics import accuracy_score
## use accuracy_score function on y_test and y_pred and print results
print('Accuracy: %.2f' % accuracy_score(y_test_log, y_pred))

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, color = 'r', marker ='o', label ='Iterations')
plt.xlabel('Epoch')
plt.ylabel('# of Misclassifications')
plt.legend(loc = 'upper right')
#pylab.savefig('perception_epoches.png',dpi=300)
plt.show()

plt.close('all')


# In[107]:




from matplotlib.colors import ListedColormap

def plot_decision_regions(X_iris_scaled,Y_values_iris, classifier, resolution = 0.02):
    # setup the markers nd colors map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(Y_values_iris))])
    print('------------------------------------------')

#plot the decision space
    x1_min, x1_max = X_iris_scaled[:, 0].min()-1, X_iris_scaled[:, 0].max()+1
    x2_min, x2_max = X_iris_scaled[:, 1].min()-1, X_iris_scaled[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class sample
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X_iris_scaled[y == cl, 0], y=X_iris_scaled[y ==cl, 1], s= 10, alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)

plt.close('all')


# In[108]:


plot_decision_regions(X_iris_scaled,Y_values_iris, classifier = ada)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc = 'upper left')
plt.title('Decision Surface', fontsize=12)
pylab.savefig('decision_surface.png',dpi=300)
plt.show()


# In[ ]:



