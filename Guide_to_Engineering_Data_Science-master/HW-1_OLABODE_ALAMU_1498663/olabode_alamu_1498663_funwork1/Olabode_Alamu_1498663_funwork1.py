
# coding: utf-8

# Olabode Alamu
# 1498663
# A guide to engineering data science 
# Fun-work 1
# 
# 1a

# In[39]:

# Data sources
# Web sources
# CSV - Comma Separated Values
# Excel
# Text files
# json files
# XML files
# binary files


# 1c Install missing packages

# In[40]:

import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# 2a Data types in python

# In[41]:

# 1 Strings
# 2 Integer
# 3 Lists
# 4 Dictionary
# 5 Tuples


# 
# 2b Basic operations
# Strings

# In[42]:

z = 'eggs'


# In[43]:

v = 'chicken'


# In[44]:

z + ' ' + v # String concatenation 


# In[45]:

z * 3 


# Integer

# In[46]:

c = 2  # Integer type
d = 4


# In[47]:

c + c  # addition


# In[48]:

c * d  # multiplication


# In[49]:

c / d # division


# In[50]:

d - c # Substraction


# List

# In[51]:

b = ['new','clothes','are','good']  # Lists
f = ['and','watches','too']
Continent = ['Europe','America','Asia']


# In[52]:

b


# In[53]:

b * 2


# In[54]:

b + f  # List concatenation


#  Dictionary

# In[55]:

dict = {'Country':['Germany','USA','Japan'], 'Capital':['Berlin', 'Washington DC', 'Tokyo']}


# In[56]:

dict


# In[57]:

dict['Capital']


# 2c

# In[58]:

df = DataFrame(data = dict, index = Continent ) # Pandas DataFrame object


# In[59]:

df


# In[60]:

df.to_csv('new.csv')


# Import data from different files can be done with the pandas library as shown below

# In[61]:

df1 = pd.read_csv('new.csv')


# In[62]:

df1


# In[63]:

# Import from excel files
df2 = pd.read_excel('p.xlsx')


# In[64]:

df2


# In[ ]:



