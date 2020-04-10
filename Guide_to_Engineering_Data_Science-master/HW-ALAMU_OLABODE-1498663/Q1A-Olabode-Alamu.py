
# coding: utf-8

# In[7]:


# Based on the built in function LA.svd


# In[1]:


import numpy as np
# insert the matrix as A

#A = np.array([[2, 4], [1, 3], [0, 0], [0, 0]])
A = np.array([[2,0,8,6,0], [1,6,0,1,7], [5,0,7,4,0], [7,0,8,5,0],[0,10,0,0,7]])
print('A array')
print(A)

from numpy import linalg as LA


# In[2]:


u,s,v = LA.svd(A, full_matrices=True)

s=np.diag(s)


# In[3]:


print(u)


# In[4]:


print(s)


# In[5]:


print(v)


# In[6]:


b=u.dot(s)
Aestimate_func = b.dot(v)
print(Aestimate_func)


# In[7]:


import numpy as np
# insert the matrix as A
A = np.array([[2,0,8,6,0], [1,6,0,1,7], [5,0,7,4,0], [7,0,8,5,0],[0,10,0,0,7]])
print('A array')
print(A)


# In[8]:


# Calculate the transpose of A
At =A.T
print('------------------------------------')
print('Transpose A array')
print(At)
print('------------------------------------')


# In[9]:


# multiply A transpose to A
AT_A = At.dot(A)
print(AT_A)


# In[10]:


from numpy import linalg as LA


# In[11]:


# Calculate the eigenvalues and eigenvectors
wv,vv = LA.eig(AT_A)
print('------------------------------------')
print('Eigen values')
print(wv)
print('------------------------------------')
print('------------------------------------')
print('The V column is')
print('Eigenvectors')
print(vv)
print('------------------------------------')


# In[13]:


print(vv.T)


# In[14]:


print(v)


# In[15]:


print('------------------------------------')
A_AT=A.dot(At)
print('Dot Product of A * Transp_A array')
print(A_AT)
print('------------------------------------')
"""
The eigenvalues and normalized unit vectors for U - right singular vector
"""
wu,vu =LA.eig(A_AT)
print('------------------------------------')
print('The eigenvalues of A*AT is: ')
print(wu)

print('------------------------------------')
print('The U column is')
print(vu)
print('------------------------------------')


# In[16]:


# calculate the transpose of V and print
#vv_rev = np.array([[vv[0,1],vv[0,0]], [vv[1,1],vv[1,0]]])
vv_rev =np.array([[vv[0,4],vv[0,3],vv[0,2],vv[0,1],vv[0,0]],
                  [vv[1,4],vv[1,3],vv[1,2],vv[1,1],vv[1,0]],
                  [vv[2,4],vv[2,3],vv[2,2],vv[2,1],vv[2,0]],
                  [vv[3,4],vv[3,3],vv[3,2],vv[3,1],vv[3,0]],
                  [vv[4,4],vv[4,3],vv[4,2],vv[4,1],vv[4,0]]])
#print(vv_rev)
V = vv_rev
Vt=V.T


# In[17]:


print(V)


# In[18]:


print(Vt)


# In[19]:


print(v)


# In[22]:


# Calculate the S matrix
S =np.zeros(A.shape) # initialises the S matrix to zero and ensures it has the same shape as A

S_list = []

# iterates through the eigenvalues 
for i in wv:
    #print(i)
    S_list.append(np.sqrt(i)) # Computes the square root of each eigen value and appends
    S_list.sort(reverse=True) # sorts the list from highest to lowest
print(S_list)
print('---------------------------------------------------------')
# computes the number of diagonals that would be filled in the S matrix
x,y = A.shape # x computes number of rows, y computes number of columns

#the lower of the two - number of rows / columnms would determine the length of diagonal in S
for i in range(min(x,y)): 
    S[0+i,i] = S_list[i]
   
print('The S matrix is: ')
print(S)
    


# In[23]:


# Estimates the A matrix based on the singular value decomposition
print('------------------------------------')
S_Vt=S.dot(v)
A_est=vu.dot(S_Vt)
print(A_est)
print('------------------------------------')

