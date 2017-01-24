import math
import matplotlib.pyplot as plt
import numpy as np

def h(x, t0, t1):
    return t1+t0*x

def J(data, t0, t1):
    j = 0
    for d in data:
        j += math.pow((h(d[0], t0, t1) - d[1]), 2)
    return j/(2*len(data))

#gradientFunction
def gradientJ(data,t0,t1):
    t0_grad = 0
    t1_grad = 1
    m = len(data)
    for d in data:
        t0_grad += d[0]*(h(d[0], t0, t1) - d[1])
        t1_grad += (h(d[0], t0, t1) - d[1])
    return t0_grad/m, t1_grad/m

#plot the given learning data
def plotData(data):
    plt.scatter(*zip(*data), marker='o')

#plot a line h(x)=t0+t1*x
def plotLine(t0, t1, x_range, color):
    x = np.array(x_range)
    y = h(x, t0, t1)
    return plt.plot(x, y, color=color)

#finds t0 and t1 for a certain number of iterations
def findParams(iterations, t0, t1):
    for i in range(iterations):
        t0_grad, t1_grad = gradientJ(data, t0, t1)
        t0 = t0 - alpha * t0_grad
        t1 = t1 - alpha * t1_grad
        print(t0, t1, J(data, t0, t1))
    return t0, t1

data = [(0,0),(1,5),(2,4), (7,9), (22,13)]
#data = [(3,2),(1,2),(0,1),(4,3)]
alpha = 0.0001
maxIter = 999

t0 = 2
t1 = 3

for iteration in range(10, maxIter):
    t0Res, t1Res = findParams(iteration, t0,t1)
    #plot the params according to the number of iterations
#    plt.scatter(iteration, t0Res)
#    plt.scatter(iteration, t1Res)
    #plot h for this t0 and t1
#    plotLine(t0Res,t1Res,range(0,25), 'blue')
    #plot final regression line    
    plotLine(t0Res, t1Res, range(0,25), 'red')

plotData(data)
plt.show()
