
# coding: utf-8

# In[365]:

"""
HOMEWORK 5 SOLUTION CREATED BY
OLABODE ALAMU
1498663
GUIDE TO ENGINEERING DATA SCIENCE

"""


# In[212]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[213]:

# Import the dataset
#head_column = ['Date','Q-E','ZN-E','PH-E','DBO-E','DQO-E','SS-E','SSV-E','SED-E','COND-E','PH-P','DBO-P','SS-P','SSV-P','SED-P','COND-P','PH-D','DBO-D','DQO-D','SS-D','SSV-D','SED-D','COND-D','PH-S','DBO-S','DQO-S','SS-S','SSV-S','SED-S','COND-S','RD-DBO-P','RD-SS-P','RD-SED-P','RD-DBO-S','RD-DQO-S','RD-DBO-G','RD-DQO-G','RD-SS-G','RD-SED-G']   

Water = pd.read_csv('water-treatment.data', header = None, na_values = '?', index_col = 0)


# In[324]:

# Prints out the top 5 rows of the dataset
print(Water.head())


# In[325]:

print(Water.shape) # Shows the shape of the dataframe


# In[215]:

X = Water.columns # This shows all the columns present in the dataset


# In[216]:

print(X)


# In[326]:

# This for loop was created to iterate through the difference columns of the dataset, split out each column
# and calculate the different statistical parameters for each column and print them out
for column in Water.columns:
    Maximum = Water[i].max()
    Minimum = Water[i].min()
    Mean_value = Water[i].mean()
    Median_value = Water[i].median()
    Standard_dev = Water[i].std()
    print('Column ',i,'\nMaximum value is ', Maximum, '\nMinimum value is ', Minimum,
          '\nMean value is ', Mean_value, '\nMedian is ', Median_value, 
          '\nStandard deviation is ', Standard_dev)


# In[327]:

# Computes the number of missing values in each column and prints out a statement
Missing_values = Water.isnull().sum()
for column in Water.columns: # iterates through each column
    Missing_value = Missing_values[column]
    print('Column ',column, ' has ',Missing_value, ' number of missing values.')


# In[328]:

# Import missing value using the median
# Step 1: Computes the medians across all the columns and passes them into a list
Value_list = [] # Creates an empty list

for column in Water.columns:
    Median_value = Water[column].median()
    Value_list.append(Median_value)
print(Value_list)

# Step 2 Transform the list of median values into a dictionary where the column number is the key and the median
# values are the value in the key: value pair of the dictionary
Median_dict = {}
for column in Water.columns:
    Median_dict[column]= Value_list[column-1]
    
print(Median_dict)

# Step 3 Fill the missing value with the medians in the dictionary

Water_cleaned = Water.fillna(value = Median_dict)

print(Water_cleaned)






    
    


# In[329]:

print(Water_cleaned)


# In[330]:

# Scale the variables
from sklearn.preprocessing import StandardScaler
scaled = StandardScaler()
scaled.fit(Water_cleaned)
Water_cleaned_scaled = scaled.transform(Water_cleaned) # Scaled dataset


# In[331]:

print(Water_cleaned_scaled)


# In[332]:

# Determine the two principal components among the features
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)  # Request that two principal components be calculated from the features
pca.fit(Water_cleaned_scaled)  # fit the dataset
pca_comps = pca.transform(Water_cleaned_scaled) # assign the principal components to a variable


# In[333]:

print(pca_comps)


# In[334]:

print(pca_comps.shape)  # SHows the new shape of the dataframe

# notice that instead of 38 columns, it has been reduced to 2 columns


# In[335]:

# Show the PCA components
print(pca.components_)


# In[336]:

# Create a visual heat map of the prinmcipal components
df_pca_comps = pd.DataFrame(data = pca.components_ , columns = np.arange(1,39))
sns.heatmap(data = df_pca_comps)
plt.figure(figsize=(40,10))
plt.show()


# In[337]:

np.arange(1,39)


# In[339]:

print(df_pca_comps)


# In[340]:

from scipy.cluster.hierarchy import dendrogram, linkage


# In[341]:

Water_matrix = np.array(Water_cleaned_scaled)


# In[342]:


# generate the linkage matrix
Z = linkage(Water_matrix, 'ward')


# In[343]:

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist



# In[344]:

c, coph_dists = cophenet(Z, pdist(Water_matrix))
print(c)


# In[345]:


# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# In[346]:

plt.scatter(pca_comps.transpose()[0], pca_comps.transpose()[1])
plt.show()


# In[348]:

print(Water_matrix.shape)


# In[349]:

# Import Kmeans
from sklearn.cluster import KMeans


# In[350]:

# Create an instance of a Kmeans model with 4 clusters
kmeans = KMeans(n_clusters = 4)


# In[351]:

# fit the data
kmeans.fit(Water_cleaned_scaled)


# In[354]:

print(kmeans.cluster_centers_.shape)


# In[355]:

print(kmeans.cluster_centers_)


# In[356]:

sns.heatmap(data = kmeans.cluster_centers_)
plt.figure(figsize=(40,10))
plt.show()


# In[357]:

C = kmeans.cluster_centers_.transpose()


# In[359]:

print(C.shape)


# In[360]:

kmeans.labels_.shape


# In[361]:

C[:,0].shape


# In[363]:

m,n = Water_matrix.shape
for i in range(n):
    #print(i)
    plt.figure()
    plt.scatter(Water_matrix[:,0],Water_matrix[:,i], c = kmeans.labels_)
    plt.show()


# In[309]:

plt.scatter(C[:,0], C[:,1], c ='r')
plt.scatter(C[:,0], C[:,2], c = 'b')
plt.scatter(C[:,0], C[:,3], c = 'g')
plt.show()


# In[364]:

print('The End')


# In[ ]:



