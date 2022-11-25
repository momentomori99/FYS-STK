import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import random


N = 100
x = np.random.rand(N,1)
#x = np.linspace(0,1,N)

y = 2.0 + 5 * x * x+0.1 * np.random.randn(N,1)

#Creating Design Matrix:
X = np.zeros([N, 3]) #3 here represent properties (p), it is 3 since we are studying second-order polynomial.
for i in range(len(x)):
    X[i][0] = x[i]**0
    X[i][1] = x[i]**1
    X[i][2] = x[i]**2


def find_Beta_OLS(X, y):
    """
    Input: Design matrix X, and datapoints in vector form y.
    Output: Estimated Beta vector (estimated coefficients for the model)
    """
    X_T = np.transpose(X) #Finding transpose of X
    XX_T = np.matmul(X_T, X) #Multiplying X and Transpose of X
    inv_XX_T = np.linalg.inv(XX_T) #Inverting the last line
    last_ = np.matmul(inv_XX_T, X_T) #Multiplying that inverse with X-Transpose
    Beta = last_.dot(y) #Multiplying all this with y-vector
    return Beta


def y_tilda(N, n_start, n_finish, Beta, y_data, x_data):
    """
    Input:
    N: number of datapoints
    n_start: startpoint of data on x-axis
    n_finish: Endpoint of data on x-axis
    Beta: the estimated coefficient array
    y: observed datapoints
    """
    x = np.linspace(n_start, n_finish, N)
    y_tilda = Beta[0][0] + Beta[1][0]*x + Beta[2][0]*(x**2)

    return y_tilda







Beta = find_Beta_OLS(X, y)
y_approx = y_tilda(100, 0, 1, Beta, y, x)

x_clean = np.linspace(0, 1, N)
plt.plot(x_clean, y_approx, "r", label="Approx. by me")
plt.plot(x, y, "o", label="Real data")
plt.legend()
plt.grid(True)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Simple Linear Regression')
plt.show()
