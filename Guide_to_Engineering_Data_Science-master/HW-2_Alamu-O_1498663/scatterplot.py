
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[11]:

# input the data set from the excel file
df = pd.read_excel('outlier_data.xlsx', sheetname = 1)


# In[16]:

X = np.array(df['X'])  # Creates an array object of the values in the X column


# In[14]:

Y = np.array(df['Y'])   # Creates an array object of the values in the Y column


# In[17]:

plt.scatter(X,Y) # Plots a simple scatter diagram
plt.show()


# In[ ]:



