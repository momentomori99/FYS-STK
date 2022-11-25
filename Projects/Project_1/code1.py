# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np
# from random import random, seed
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# #The function itself:
# def FrankeFunction(x,y):
#     term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
#     term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
#     term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
#     term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
#     return term1 + term2 + term3 + term4
#
# # Make data.
# x = np.arange(0, 1, 0.05)
# y = np.arange(0, 1, 0.05)
# x, y = np.meshgrid(x,y)
# z = FrankeFunction(x, y)
#
# # Plot the surface.
# surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # Customize the z axis.
# ax.set_zlim(-0.10, 1.40)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()
#
#
# #Try to make approx
# X = np.zeros([len(x), 8])
# x_r = x.ravel()
# y_r = y.ravel()
# X[:,0] = 1
# X[:,1] = x_r
# X[:,2] = y_r
# X[:,3] = x_r**2
# X[:,4] = y_r**2
# X[:,5] = x_r*y_r
# X[:,6] = x_r**3
# X[:,7] = y_r**3
# X[:,8] = (x_r**2)*(y_r**2)
# X[:,9] = x_r**4
# X[:,0] = y_r**4
#
# X_T = np.transpose(X)
# XX_T = np.matmul(X_T, X)
# inv_XX_T = np.linalg.inv(XX_T)
# last_ = np.matmul(inv_XX_T, X_T)
#
# Beta = last_.dot(z)
# Common imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer



def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


# Making meshgrid of datapoints and compute Franke's function
n = 5
N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
z = FrankeFunction(x, y)
X = create_X(x, y, n=n)
# split in training and test data
X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.2)
