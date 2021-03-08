#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 10:36:15 2021

@author: Ingrid
"""

import numpy as np
from matplotlib import pyplot as plt 

def relu(x):
    return max(0,x)

def computeSquare(x, nrHiddenLayers):
    """Compute square of x using neural network with m hidden layers.
    
    L = m + 2
    W = 3m + 2
    E = 12m - 5
    """
    b1 = 0
    b2 = -1/2
    b3 = -1
    w1 = 2
    w2 = -4
    w3 = 2
    
    output = x
    f = x
    for i in range(1,nrHiddenLayers+1):
        n1 = relu(f + b1)
        n2 = relu(f + b2)
        n3 = relu(f + b3)
        
        f = w1*n1 + w2*n2 + w3*n3
        
        output += -1/(2**(2*i))*f
        
    return output
vcomputeSquare = np.vectorize(computeSquare)

epsilon = 0.05
m = np.ceil(1/2*np.log2(1/epsilon) - 1).astype('int')
x = 0.4
xSquaredEst = computeSquare(x,m)
print("x = ", x, ", layers = ",m, ", estimate = ", xSquaredEst, ", real value = ", x**2, "Error = ", np.abs(xSquaredEst - x**2))



def plotfm(m):
    """Create plot similar to Yarotsky2017 Figure 2b""" 
    fig = plt.figure(figsize =(10, 7)) 
    for j in range(0,m):
        n = np.repeat(j, 100)
        x = np.arange(0,1,0.01)
        xSquaredEst = vcomputeSquare(x, n)
        plt.plot(x, xSquaredEst, label = j)
       
    plt.xlabel("$x$")
    plt.ylabel("$f_m(x)$")
    plt.legend(loc = 'best')#prop={'size': 16})
    
    # plt.savefig("functionfm.pdf", bbox_inches = 'tight')
    
# plotfm(4)