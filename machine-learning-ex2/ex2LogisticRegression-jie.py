 # coding=utf8
__author__ = 'Jie'

import numpy as np
from matplotlib import pyplot as pp
import math as mm
import numpy.linalg  as alg
import os
import scipy.optimize as op

def plotData(X,y):
    pos=[index for index, value in enumerate(y) if value==1]
    neg=[index for index, value in enumerate(y) if value==0]

    fig=pp.figure()
    ax=pp.subplot(111)
    ax.plot(X[pos,0],X[pos,1],'k+',label='Admitted')
    ax.plot(X[neg,0],X[neg,1],'ko',label='No admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    pp.show()

def sigmoid(z):  # z is set to be a column vector
    g=np.zeros(np.shape(z),'d') # get the same size of matrix/vector with z
    #m1=np.shape(g)[0]
    #n1=np.shape(g)[1]
    g=1/(1+np.exp(z))
    return g

'''
    for i in range(0,m1):
        for j in range(0,n1):
            g[i,j]=1/(1+np.exp(-z[i,j]))
    return g
'''
def costFunction(theta,X,y):
    m=len(y)
    J=0
    grad=np.zeros(np.shape(theta),'d')
    z=np.dot(X,theta)
    g=sigmoid(z)
    predictions=-y*np.log(g)-(1-y)*np.log(1-g)
    J=(1/m)*np.sum(predictions)


    for i in range(0,np.shape(theta)[0]):
        grad[i,0]=(1/m)*np.sum((g-y).T*X[:,i])  # be careful the vector multiplication.  transpose is needed. 　g-y is a array, X[:,i] is a vector!!
        # so we have to keep all the type of data the same
    return J,grad


def gradient(theta,X,y):
    m=len(y)
    J=0
    grad=np.zeros(np.shape(theta),'d')
    z=np.dot(X,theta)
    g=sigmoid(z)
    predictions=-y*np.log(g)-(1-y)*np.log(1-g)
    J=(1/m)*np.sum(predictions)

    grad=(float(1)/m)*((g-y).T.dot(X))
    return  grad

def main():
    data=open('ex2data1.txt').readlines()
    X1=[float(elem.split(',')[0]) for elem in data]
    X2=[float(elem.split(',')[1]) for elem in data]
    y=[int (elem.split(',')[2].strip()) for elem in data]
    X=np.c_[X1,X2]
    m=len(y)
    print ('the scores sample X is: \n', X)
    print ('the admission decision for each sample is: \n',y)

    plotData(X,y)
    one=np.ones((m,1),'d')
    X=np.c_[one,X]
    n=np.shape(X)[1]
    initial_theta=np.zeros((n,1),'d')
    y=np.asarray(y)
    y.shape=(m,1)
    cost1=costFunction(initial_theta,X,y)
    gradient1=gradient(initial_theta,X,y)
    print ('the cost1 at initial theta is: \n',cost1)
    print('the gradient at initial theta (zeros) is: \n', gradient1)

    #Result=op.minimize(fun=costFunction,x0=initial_theta,args=(X,y),method='TNC',jac=gradient)
    #optimal_theta=Result.X
    #print(op.fmin_bfgs(f, initial_theta, fprime, disp=True, maxiter=400, full_output = True, retall=True))


if __name__=='__main__':
    main()



