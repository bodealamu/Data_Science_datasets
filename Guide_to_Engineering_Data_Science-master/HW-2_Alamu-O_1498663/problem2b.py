
# coding: utf-8

# In[11]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


# In[12]:

# input the data set from the excel file
df = pd.read_excel('outlier_data.xlsx', sheetname = 1)


# In[13]:

df


# In[14]:

X = np.array(df['X'])  # Creates an array object of the values in the X column
Y = np.array(df['Y'])   # Creates an array object of the values in the Y column
C = list(zip(X,Y)) # This pairs up the X and Y corresponding values and creates a new combined list


# In[15]:

Xdata = np.array(C) # Converts the combined list to an array


# In[17]:

Xdata


# In[18]:

#knn function gets the dataset and calculates K-Nearest neighbors and distances
def knn(df,k):
    nbrs = NearestNeighbors(n_neighbors=3)
    nbrs.fit(df)
    distances, indices = nbrs.kneighbors(df)
    return distances, indices

#reachDist calculates the reach distance of each point to MinPts around it
def reachDist(df,MinPts,knnDist):
    nbrs = NearestNeighbors(n_neighbors=MinPts)
    nbrs.fit(df)
    distancesMinPts, indicesMinPts = nbrs.kneighbors(df)
    distancesMinPts[:,0] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,1] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,2] = np.amax(distancesMinPts,axis=1)
    return distancesMinPts, indicesMinPts

#lrd calculates the Local Reachability Density
def lrd(MinPts,knnDistMinPts):
    return (MinPts/np.sum(knnDistMinPts,axis=1))

#Finally lof calculates lot outlier scores
def lof(Ird,MinPts,dsts):
    lof=[]
    for item in dsts:
       tempIrd = np.divide(Ird[item[1:]],Ird[item[0]])
       lof.append(tempIrd.sum()/MinPts)
    return lof

#We flag anything with outlier score greater than 1.2 as outlier#This is just for charting purposes
def returnFlag(x):
    if x['Score']>1.2:
       return 1
    else:
       return 0


# In[19]:

knn(Xdata,2)


# In[21]:

reachDist(Xdata,2)


# In[ ]:



