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

def square01(x, nrHiddenLayers):
    """Compute square of x in [0,1] using neural network with nrHiddenLayers hidden layers.
    
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
    g = x
    for i in range(1,nrHiddenLayers+1):
        n1 = relu(g + b1)
        n2 = relu(g + b2)
        n3 = relu(g + b3)
        
        g = w1*n1 + w2*n2 + w3*n3
        
        output += -1/(2**(2*i))*g
        
    return output

vSquare01 =  np.vectorize(square01)

def squareM(x, M, nrHiddenLayers):
    """Compute square of x in [-M,M] using neural network with nrHiddenLayers hidden layers. """
    return square01(np.abs(x)/M, nrHiddenLayers)*M
vSquareM =  np.vectorize(squareM)

def mult2(x, y, M, N, nrHiddenLayers):
    # delta = errorBound/(6*(M*N)**2)
    return 2*(M*N)**2*(square01(np.abs(x+y)/(2*M*N), nrHiddenLayers) - square01(np.abs(x)/(2*M*N), nrHiddenLayers) - square01(np.abs(y)/(2*M*N), nrHiddenLayers))
vMult2 = np.vectorize(mult2)

epsilon = 0.05
m = np.ceil(1/2*np.log2(1/epsilon) - 1).astype('int')
x = 0.4
xSquaredEst = square01(x,m)
print("x = ", x, ", layers = ",m, ", estimate = ", xSquaredEst, ", real value = ", x**2, "Error = ", np.abs(xSquaredEst - x**2))



def plotSquare01(m):
    """Create plot similar to Yarotsky2017 Figure 2b""" 
    fig = plt.figure(figsize =(10, 7)) 
    for j in range(0,m):
        n = np.repeat(j, 100)
        x = np.arange(0,1,0.01)
        xSquaredEst = vSquare01(x, n)
        plt.plot(x, xSquaredEst, label = j)
       
    plt.xlabel("$x$")
    plt.ylabel("$f_m(x)$")
    plt.legend(loc = "best")#prop={'size': 16})
    
    # plt.savefig("functionfm.pdf", bbox_inches = 'tight')
    
def plotSquareM(m,M):
    fig = plt.figure(figsize =(10, 7)) 
    for j in range(0,m):
        n = np.repeat(j, 100)
        vM = np.repeat(M, 100)
        x = np.arange(-M,M,2*M/100)
        xSquaredEst = vSquareM(x, vM, n)
        plt.plot(x, xSquaredEst, label = j)
       
    plt.xlabel("$x$")
    plt.ylabel("$\tilde{f}_m(x)$")
    plt.legend(loc = "best")#prop={'size': 16})
    
    # plt.savefig("functionfm.pdf", bbox_inches = 'tight')
    
def plotMult2(m,x,N):
    fig = plt.figure(figsize =(10, 7)) 
    M = np.abs(x)
    for j in range(0,m):
        n = np.repeat(j, 1000)
        vM = np.repeat(M, 1000)
        vN = np.repeat(N, 1000)
        vx = np.repeat(x, 1000)
        vy = np.arange(-N,N,2*N/1000)
        xyEst = vMult2(vx, vy, vM, vN, n)
        plt.plot(vy, xyEst, label = j)
       
    plt.xlabel("$y$")
    plt.ylabel("$xy$")
    plt.legend(loc = "best")#prop={'size': 16})
    
    # plt.savefig("functionfm.pdf", bbox_inches = 'tight')
    
# plotSquare01(4)
# plotSquareM(5,5)
plotMult2(7,-2,50)

# fig = plt.figure(figsize =(10, 7)) 
# x = np.arange(0,2,0.01)
# y = np.arange(.5,1,0.01)
# z = np.arange(1,2,0.01)
# plt.plot(x, 2*x, label = "2$\sigma(x)$")
# plt.plot(y, 2*y-4*(y-1/2), label = "2$\sigma(x)-4\sigma(x-1/2)$")
# plt.plot(z, 2*z-4*(z-1/2)+2*(z-1), label = "2$\sigma(x)$")
# plt.legend(loc = "best")#prop={'size': 16})

