
# coding: utf-8

# In[1]:

def collatz(number):
    """
    Olabode Alamu
    1498663
    Guide to Engineering Data Science HW 3
    THis function takes in an interger or a float number and applys the collatz routine to it.
    """
    try: # Used to handle exceptions gracefully
        if number % 2 == 0:  # CHecks for even number
            print(number / 2)
            return (number / 2)
        else:
            print(3*number + 1)
            return (3*number + 1)
        
    except:
        print('The parameter is expected to be an integer or float')


# In[2]:

# Handles floats
collatz(5.0)


# In[3]:

# Handles intergers
collatz(10)


# In[5]:

# Handles wrong types of inputs - lists, strings etc gracefully 
collatz('Olabode Alamu')


# In[6]:

collatz([1,2,3])


# In[ ]:



