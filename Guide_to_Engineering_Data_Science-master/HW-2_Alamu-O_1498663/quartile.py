
# coding: utf-8

# In[14]:

def quartile(data):
    """
    Determines the first, second and third quartiles using the tukey hinges method
    """
    global second_quartile, first_quartile, third_quartile, IQR
    import numpy as np
    data.sort()  # Sorts the data in the increasing order
    print('The data is sorted in increasing order',data)
    
    
    length = len(data) # Calculates the lenght of the data
    print('Number of entries in dataset: ', length)
    
    second_quartile = np.median(data)  # Calculates the median of the data
    print('The median is ', second_quartile)
    
    
    
    if length % 2 == 0:
        mid = length / 2  # Calculates the midpoint of the data set
        first_half = data[ : int(mid)]
        print('The first section of the dataset: ',first_half)
        first_quartile = np.median(first_half)
        
        print('The first quartile is', first_quartile)
        
        second_half = data[int(mid):] # Sections off the second part of the data after the median
        print('The second part of the data is',second_half)
        third_quartile = np.median(second_half)
        
        print('The third quartile is', third_quartile)
        
        IQR = third_quartile - first_quartile  # Calculates the interquartile range
        print('The interquartile range is :', IQR)
        
        
    else:
        position_median = data.index(second_quartile) # finds the position of the median in the dataset
        first_half = data[ : position_median + 1] # sections off the first part
        print('The first section of the dataset: ',first_half)
        first_quartile = np.median(first_half) # Calculates the first quartile
        print('The first quartile is', first_quartile)
        
        second_half = data[position_median:] # Sections off the second part of the data after the median
        print('The second part of the data is',second_half)
        third_quartile = np.median(second_half)  # Calculates the third quartile
        
        print('The third quartile is', third_quartile)
        
        IQR = third_quartile - first_quartile  # Calculates the interquartile range
        print('The interquartile range is :', IQR)
        
    lower_bound = first_quartile - (1.5*IQR)
    
    print('=========================================================================================')
    print('The lower bound is: ',lower_bound)
    
    upper_bound = third_quartile + (1.5*IQR)
    print('The upper bound is: ',upper_bound)
    
    print('=========================================================================================')
    
    for i in data:
        if i < lower_bound or i > upper_bound:
            print('Outlier present')
            print(i)
            break
            
    else:
        print('No outlier present in dataset')
    


# In[ ]:



