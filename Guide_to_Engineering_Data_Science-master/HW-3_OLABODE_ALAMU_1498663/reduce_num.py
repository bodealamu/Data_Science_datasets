
# coding: utf-8

# In[1]:

def reduce_num():
    """
    Olabode Alamu 1498663
    Guide to Engineering Data Science
    Function written to request user input, call the collatz function 
    and iterate till the inputted number is reduced to 1.
    """
    try:
        num = float(input('Enter a number here: ')) # COnverts the input from a string to a float
        v = collatz(num) # Calls on the collatz function to initialise a value
        count = 1  # This initialises the number of times collatz was called
    
        while v > 1:  # The while loop interates till the condition is met
            v=collatz(v)
            count = count + 1
        
        print('The number has been reduced to',v)
        print('This was done after calling collatz ', count,' number of times.' )
        
    except:
        print('A number or float was expected as input.')


# In[2]:

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


# In[3]:

reduce_num()


# In[4]:

reduce_num()


# In[5]:

reduce_num()


# In[ ]:



