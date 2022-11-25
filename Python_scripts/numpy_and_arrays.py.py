import numpy as np

"""
Arrays/Vectors
"""
n = 10
x0 = np.random.normal(size=n)

x1 = np.log(np.array([4,7,8])) #Python intrepers these number as integers
x2 = np.log(np.array([4,7,8], dtype = np.float64)) #Here we change it so it intrepers them as floats
x3 = np.log(np.array([4.0,7.0,8.0])) #Or you can just write them as this
#print(x3.itemsize) #Shows how many bits it holds


"""
Matrices
"""

A0 = np.array([ [4.0, 7.0, 8.0], [3.0, 10.0, 11.0], [4.0, 5.0, 7.0] ])
#print(A0.shape) #Shows that this is 3x3 matrix
#print(A0)
#print(A0[:,0]) #This prints out the first column of the matrix
#print(A0[0,:]) #This prints out the first row of the matrix

A1 = np.zeros([n,n]) #Makes a nxn matrix that contains only zeros
A2 = np.ones([n,n]) #Makes a nxn matrix that contains only ones
A3 = np.random.rand(n,n) # #Makes a nxn matrix that contains random numbers
#print(A3)


"""
Covariant elements, and eigenvalues with eigenvectors
"""

n = 3
x = np.random.normal(size=n)
#print(f"mean value of x: {np.mean(x)}")
y = 4 + 3*x + np.random.normal(size=n)
#print(f"mean value of y: {np.mean(y)}")
z = x**3+np.random.normal(size=n)
#print(f"mean value of z: {np.mean(z)}")

W = np.vstack([x, y, z]) #Stacks up arrays into a matrix
#print(W)
Sigma = np.cov(W)
#print(Sigma)

eigenvalues, eigenvectors = np.linalg.eig(Sigma)
# print(eigenvalues)
# print(eigenvectors)

import matplotlib.pyplot as plt
from scipy import sparse

eye = np.eye(4) #makes 4x4 identity matrix
#print(eye)

sparse_mtx = sparse.csr_matrix(eye) #Don't know what 
print(sparse_mtx)
