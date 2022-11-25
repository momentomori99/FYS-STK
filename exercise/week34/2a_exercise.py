import numpy as np
import matplotlib.pyplot as plt


N = 100
x = np.random.rand(N,1)
y = 5.0+10*x*x+2*np.random.randn(N,1)



#We need to build X matrix (design matrix)

X = np.zeros([N, 3])
for i in range(len(x)):
    X[i][0] = x[i]**0
    X[i][1] = x[i]**1
    X[i][2] = x[i]**2
#Now we need to find B with it
X_T = np.transpose(X)
XX_T = np.matmul(X_T, X)
inv_XX_T = np.linalg.inv(XX_T)
last_ = np.matmul(inv_XX_T, X_T)
Beta = last_.dot(y)

x_clean = np.linspace(0, 1, 100)
def y_calc(x):
    return Beta[0][0] + Beta[1][0]*x + Beta[2][0]*(x**2)

y_aprox = np.zeros(100)
for i in range(len(x)):
    y_aprox[i] = y_calc(x[i])

print(y_aprox.shape)
print(x_clean.shape)

#Error
def MSE(y, y_aprox):
    n = np.size(y)
    return np.sum((y-y_aprox)**2)/n
#print(MSE(y, y_aprox))




plt.plot(x, y, "bo")
plt.plot(x_clean, y_calc(x_clean), "r")
plt.show()
