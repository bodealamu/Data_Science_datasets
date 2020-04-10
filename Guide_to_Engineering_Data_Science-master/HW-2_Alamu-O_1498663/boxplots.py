
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:

# Define the datasets
X = [61, 10, 32, 19, 22, 5, 100, 29, 36, 14, 49, 3]
Y = [2.3, 2.7, 1.7, 1.9, 2.1, 2.8, 1.8, 2.4, 5.9]


# In[4]:

plt.boxplot(X,0,'gD', whis = 1.5)
plt.ylabel('Values')
plt.title('Boxplot for X')
plt.grid(which = 'both', axis = 'both')
plt.show()


# In[5]:

plt.boxplot(Y,0,'gD', whis = 1.5)
plt.ylabel('Values')
plt.title('Boxplot for Y')
plt.grid(which = 'both', axis = 'both')
plt.show()


# In[ ]:




# In[ ]:



