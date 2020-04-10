
# coding: utf-8

# In[222]:

"""
cODE WRITTEN FOR hOMEWORK 4
AUTHOR: OLABODE AFOLABI ALAMU
PEOPLESOFT ID: 1498663
GUIDE TO ENGINEERING DATA SCIENCE
27TH SEPTEMBER 2017

"""


# In[223]:

# Import the relevant libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# In[224]:

Air_quality = pd.read_excel('HW4.xlsx') # Reads the excel spreadsheet which contains the dataset


# In[226]:

Air_quality.head()


# In[227]:

NanValue = Air_quality.isnull().sum() # COmputes the number of rows that have a nan value


# In[228]:

dfNanValue = pd.DataFrame(data=NanValue, columns=[1])


# In[229]:

dfNanValue[1] # SHows the number of nan values per column


# In[230]:

dfNanValue[1].plot(kind = 'bar')    # Creates a bar chart of the number of missing values per column
plt.title('Bar chart showing the number of missing values in each column')
plt.grid()
plt.show()


# In[162]:

# Create a scatter plot of Ozone concentration against each of the other parameters


# In[231]:

plt.scatter(Air_quality['Solar.R'],Air_quality['Ozone'],c = 'r')
plt.xlabel('Solar Radiation')
plt.ylabel('Ozone Concentration')
plt.title('Plot of Ozone Concentration vs Solar radiation')
plt.grid()
plt.show()


# In[232]:

plt.scatter(Air_quality['Wind'],Air_quality['Ozone'],c = 'y')
plt.ylabel('Ozone concentration')
plt.xlabel('Wind')
plt.title('Plot of Ozone Concentration vs Wind')
plt.show()


# In[165]:

plt.scatter(Air_quality['Temp'],Air_quality['Ozone'])
plt.grid()
plt.ylabel('Ozone concentration')
plt.xlabel('Temperature')
plt.title('Plot of Ozone Concentration vs Temperature')
plt.show()


# In[233]:

# Create a scatter plot of Ozone concentration against all the variables
plt.scatter(Air_quality['Temp'], Air_quality['Ozone'], label='Temperature')
plt.scatter(Air_quality['Solar.R'],Air_quality['Ozone'],c = 'r',label = 'Solar radiation')
plt.scatter(Air_quality['Wind'],Air_quality['Ozone'],c = 'y', label = 'Wind')
plt.ylabel('Ozone concentration')
plt.legend()
plt.show()


# In[245]:

Air_quality = Air_quality.dropna() # drops all the rows which have a missing value


# In[246]:

Air_quality


# In[267]:

Air_quality.isnull().sum() # All the rows with missing values have been removed


# In[268]:

# Create a scatter plot of Ozone concentration against SOlar radiation
plt.scatter(Air_quality['Temp'], Air_quality['Ozone'], label='Temperature',marker ='+')
plt.scatter(Air_quality['Solar.R'],Air_quality['Ozone'],c = 'r',label = 'Solar radiation',marker ='+')
plt.scatter(Air_quality['Wind'],Air_quality['Ozone'],c = 'y', label = 'Wind',marker ='+')
plt.ylabel('Ozone concentration')
plt.legend()
plt.show()


# In[269]:

plt.scatter( Air_quality['Solar.R'],Air_quality['Ozone'],c = 'r', marker ='+')
plt.ylabel('Ozone concentration')
plt.xlabel('Solar Radiation')
plt.title('Plot of Ozone Concentration vs Solar radiation')
plt.grid()
plt.show()


# In[270]:

plt.scatter(Air_quality['Wind'],Air_quality['Ozone'],c = 'y',marker ='+')
plt.ylabel('Ozone concentration')
plt.xlabel('Wind')
plt.title('Plot of Ozone Concentration vs Wind')
plt.grid()
plt.show()


# In[271]:

plt.scatter(Air_quality['Temp'],Air_quality['Ozone'],marker ='+')
plt.ylabel('Ozone concentration')
plt.xlabel('Temperature')
plt.title('Plot of Ozone Concentration vs Temperature')
plt.grid()
plt.show()


# In[272]:

# Import the dataset afresh
Air = pd.read_excel('HW4.xlsx')


# In[273]:

Air.head() #Shows only the top 5 values


# In[274]:

# Calculate the mean values for each column
Ozone_mean = Air['Ozone'].mean()
Solar_mean = Air['Solar.R'].mean()
Wind_mean = Air['Wind'].mean()
Temp_mean = Air['Temp'].mean()


# In[275]:

Ozone_mean


# In[276]:

Solar_mean


# In[277]:

Wind_mean


# In[278]:

Temp_mean


# In[279]:

# Next fill each corresponding column with the corresponding mean value


# In[280]:

Air.fillna(value = {"Ozone": Ozone_mean, 'Solar.R': Solar_mean, 'Wind': Wind_mean, 'Temp': Temp_mean }, inplace = True)


# In[281]:

Air.head() # Notice the changes in the row with index 5


# In[282]:

Air.isnull().sum()  # SUms up the number of rows with a Nan value


# In[283]:

plt.scatter( Air['Solar.R'],Air['Ozone'],c = 'r', marker ='o')
plt.ylabel('Ozone concentration')
plt.xlabel('Solar Radiation')
plt.title('Plot of Ozone Concentration vs Solar radiation with mean filled values')
plt.grid()
plt.show()


# In[284]:

plt.scatter(Air['Wind'],Air['Ozone'],c = 'y',marker ='o')
plt.ylabel('Ozone concentration')
plt.xlabel('Wind')
plt.title('Plot of Ozone Concentration vs Wind with mean filled values')
plt.grid()
plt.show()


# In[285]:

plt.scatter(Air['Temp'],Air['Ozone'],marker ='o')
plt.ylabel('Ozone concentration')
plt.xlabel('Temperature')
plt.title('Plot of Ozone Concentration vs Temperature with mean filled values')
plt.grid()
plt.show()


# In[ ]:




# In[286]:

print('This is the end')
print("bows")


# In[ ]:



