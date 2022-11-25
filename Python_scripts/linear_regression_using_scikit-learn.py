import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error

x = np.random.rand(100,1)
y = 2* + 5*x+0.5*np.random.randn(100,1)

linreg = LinearRegression()
linreg.fit(x,y)

ypredict = linreg.predict(x)

print('The intercept alpha: \n', linreg.intercept_) #finding the intercept point
print('Coefficient beta : \n', linreg.coef_) #finding stigningstall
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y, ypredict))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, ypredict))
# Mean squared log error
print('Mean squared log error: %.2f' % mean_squared_log_error(y, ypredict) )
# Mean absolute error
print('Mean absolute error: %.2f' % mean_absolute_error(y, ypredict))

#Now we wish to plot them

plt.plot(x, ypredict, "g-")
plt.plot(x, y, "ro")
plt.plot(x, np.abs(ypredict-y)/abs(y), "bo")
plt.axis([0,1.0,0, 5.0])
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Simple Linear Regression')
plt.show()
