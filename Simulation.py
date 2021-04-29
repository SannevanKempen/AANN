#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 10:36:15 2021

@author: Ingrid
"""
import numpy as np
from matplotlib import pyplot as plt 
directory = "/Users/Ingrid/Documents/Github/AANN/"

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
    """Compute multiplication xy in [-M,M]x[-N,N] using neural network with nrHiddenLayers hidden layers. """
    return 2*(M*N)**2*(square01(np.abs(x+y)/(2*M*N), nrHiddenLayers) - square01(np.abs(x)/(2*M*N), nrHiddenLayers) - square01(np.abs(y)/(2*M*N), nrHiddenLayers))
vMult2 = np.vectorize(mult2)

def g(s,x):
    for k in range(0,2**(s-1)+2):
        if 2*k/2**s <= x <= (2*k+1)/2**s:
            return x*2**s - 2*k
        elif (2*k+1)/2**s <= x <= (2*k+2)/2**s:
            return 2*(k+1) - x*2**s
vg = np.vectorize(g)

def plotg(s):
    fig = plt.figure(figsize =(10, 7)) 
    for j in range(1,s+1):
        n = np.repeat(j, 1000)
        x = np.arange(0,1,0.001)
        gsx = vg(n,x)
        plt.plot(x, gsx, label = f"$g_{j}$", linewidth = 2)
       
    # plt.xlabel("$x$")
    plt.legend(loc = "best", prop={'size': 15})
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig(directory + "functiongs.pdf", bbox_inches = 'tight')
    plt.show()


def plotSquare01(m):
    """Create plot similar to Yarotsky2017 Figure 2b""" 
    fig = plt.figure(figsize =(10, 7)) 
    for j in range(0,m+1):
        n = np.repeat(j, 100)
        x = np.arange(0,1,0.01)
        xSquaredEst = vSquare01(x, n)
        plt.plot(x, xSquaredEst, label = f"$h_{j}$", linewidth = 2)
    plt.plot(x, x**2, "--", label = "$f$", color = "black")
       
    # plt.xlabel("$x$")
    plt.legend(loc = "best", prop={'size': 15})
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig(directory + "functionhm.pdf", bbox_inches = 'tight')
    
def plotSquareM(m,M):
    fig = plt.figure(figsize =(10, 7)) 
    for j in range(0,m+1):
        n = np.repeat(j, 100)
        vM = np.repeat(M, 100)
        x = np.arange(-M,M,2*M/100)
        xSquaredEst = vSquareM(x, vM, n)
        plt.plot(x, xSquaredEst, label = j)
       
    plt.xlabel("$x$", fontsize = 14)
    plt.ylabel("$\tilde{f}_m(x)$", fontsize = 14)
    plt.legend(loc = "best", prop={'size': 15})
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig(directory + "functiontildeh.pdf", bbox_inches = 'tight')
    
def plotMult2(m,x,N):
    fig = plt.figure(figsize =(10, 7)) 
    axes = plt.gca()
    axes.set_ylim([-150,100])
    M = np.abs(x)
    for j in range(1,m+1):
        n = np.repeat(j, 1000)
        vM = np.repeat(M, 1000)
        vN = np.repeat(N, 1000)
        vx = np.repeat(x, 1000)
        vy = np.arange(-N,N,2*N/1000)
        xyEst = vMult2(vx, vy, vM, vN, n)
        plt.plot(vy, xyEst, label = j)
      
    plt.plot(vy, x*vy, "--", label = "$xy$", linewidth = 2, color = "black")
    plt.xlabel("$y$", fontsize = 14)
    plt.legend(loc = "best", prop={'size': 15})
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig(directory + "functiontimes.pdf", bbox_inches = 'tight')
    
# plotg(3)
# plotSquare01(2)
# plotSquareM(5,5)
plotMult2(5,5,10)

# fig = plt.figure(figsize =(10, 7)) 
# x = np.arange(0,2,0.01)
# y = np.arange(.5,1,0.01)
# z = np.arange(1,2,0.01)
# plt.plot(x, 2*x, label = "2$\sigma(x)$")
# plt.plot(y, 2*y-4*(y-1/2), label = "2$\sigma(x)-4\sigma(x-1/2)$")
# plt.plot(z, 2*z-4*(z-1/2)+2*(z-1), label = "2$\sigma(x)$")
# plt.legend(loc = "best")#prop={'size': 16})

