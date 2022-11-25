import numpy as np
from numpy.random import randint, randn
import time

def jackknife(data, stat):
    n = len(data)
    t = np.zeros(n)
    inds = np.arange(n)
    t0 = time.time()
    for i in range(n):
        t[i] = stat(np.delete(data,i))
        #This for loop, it calls stat function, which just find the mean
        #value of the whole set of datapoints, except for one i point. It
        #repeats this n times, and stores all mean values to t array.

    # analysis
    print("Runtime: %g sec" % (time.time()-t0)); print("Jackknife Statistics :")
    print("original           bias      std. error")
    print("%8g %14g %15g" % (stat(data),(n-1)*np.mean(t)/n, (n*np.var(t))**.5))

    print(t)

    return t

def stat(data):
    return np.mean(data)

mu = 100
sigma = 15
N = 10000

x = mu + sigma*np.random.randn(N)

t = jackknife(x, stat)
