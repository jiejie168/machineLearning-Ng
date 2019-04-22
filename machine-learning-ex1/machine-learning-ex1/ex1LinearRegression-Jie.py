 # coding=utf8
__author__ = 'Jie'
'''
this is the scipt for linear regression for multiple variables.
'''

import numpy as np
from matplotlib import pyplot as pp
import math as mm
import numpy.linalg  as alg
import os
from pylab import *


def featureNormalize(X):
    X_norm=X
    n=X.shape[1]
    mu=np.zeros((1,X.shape[1]),'d')  # 1*(n), the extra column (x=1) has not been added.
    sigma=np.zeros((1,X.shape[1]),'d')


    for i in range(0,n):
        mu[0,i]= np.mean(X[:,i])  # the mean value of each feature,
        sigma[0,i]=np.std(X[:,i],ddof=1) # the standard deviation of each feature
        X_norm[:,i]=(X[:,i]-mu[0,i])/sigma[0,i]
    return X_norm, mu, sigma


def computeCostMulti(X,y,theta):
    #X is the "design matrix" containing our training examples.
    # y is the class labels
    #m=np.shape(X)[0]  # number of training number
    predictions=np.dot(X,theta)  # predictions of hypothesis on all m examples.m*(n+1), (n+1)*1
    sqrErrors=(predictions-y)*(predictions-y) # squared errors
    m=len(y)
    J=1/(2*m)*np.sum(sqrErrors)
    return J

def gradientDescentMulti(X,y,theta,alpha,num_iters):
    # this is the function to perform gradient descent to learn
    #theta
    m=len(y)
    J_history=np.zeros((num_iters,1),'d')
    #deriv1=np.zeros((m,X.shape[1]),'d')
    tempo1=theta  # a temporary array equals theta.

    for iter in range(0,num_iters):
        deriv0=np.dot(X,theta)-y

        for i in range (0,X.shape[1]):
            tempo1[i,0]=np.sum(deriv0.T*X[:, i]) # be careful the internal multiplication. a transpose is needed.

        theta=theta-alpha*(1/m)*tempo1
        J_history[iter]=computeCostMulti(X,y,theta)
    return theta, J_history

def normalEqn(X,y):
    #theta=np.zeros((X.shape[1],1),'d')
    XTX=np.dot((np.transpose(X)),X)
    inv_XTX=alg.inv(XTX)
    inv_XTXXT=np.dot(inv_XTX,np.transpose(X))
    theta=np.dot(inv_XTXXT,y)
    return theta

def predictiPrice(theta,eg1,mu,sigma): # eg1 is the input features, should be an array. 1*(n+1)
    eg1=eg1[1:]
    eg1=(eg1-mu)/sigma

    m=eg1.shape[0]
    one=np.ones((m,1),'d')
    eg1=np.c_[one,eg1]
    pr=np.dot(eg1,theta)
    return pr


def main():
    ###  for normal equation, there is no need to do the feature scaling. an extra column of 1 to the X matrix is also needed.
    data=open('ex1data2.txt','r').readlines()
    X1= [float(elem.split(',')[0]) for elem in data]
    X2=[float(elem.split(',')[1]) for elem in data]
    y=[float(elem.split(',')[2]) for elem in data]
    X=np.c_[X1,X2]  # X is already an array.
    y=np.asarray(y) # change list to array
    m=len(y)  # the number of traning sample
    one=np.ones((m,1),'d')
    X=np.c_[one,X]  # X is an array  m*(n+1),  no feature scaling

    print ('calculate theta via normal equation')
    thetaNormEqn=normalEqn(X,y)
    print('......done\n')
    print ('the theta from normal equation is: \n',thetaNormEqn)


    ## calculate the scaling features. here there is no need to add an extra 1 column for X.
    data=open('ex1data2.txt','r').readlines()
    X1= [float(elem.split(',')[0]) for elem in data]
    X2=[float(elem.split(',')[1]) for elem in data]
    y=[float(elem.split(',')[2]) for elem in data]
    X=np.c_[X1,X2]  # X is already an array.
    y=np.asarray(y) # change list to array
    y.shape=(m,1) # change vector y to m*1 array
    n=X.shape[1]  # the number of features


    features=featureNormalize(X)
    print ('the normalized X is: \n',features[0])
    print('the mean value mu of each feature is: \n',features[1])
    print('the standard deviation signa of each feature is: \n ',features[2])

    ####################### calculate the cost function and the theta using gradient discent method.
    X=np.c_[one,X]  # X is an array  m*(n+1)
    alpha=0.01 # learning rate
    numIters=4000  # interation number
    initial_theta=np.zeros((n+1,1),'d')  #(n+1)*1

    print('calculate the cost function')
    gradD=gradientDescentMulti(X,y,initial_theta,alpha,numIters)
    print ('.....done\n')
    print('the calculated theta from gradient descent is:\n',gradD[0])
    print ('the calcualted costHistory is: \n',gradD[1][0:5])


    #prediction of house prices
    print('predicting the price of the following house\n')
    eg1=[1,1650,3]
    eg1=np.asarray(eg1)
    print('sq ft is: , No of bedrooms is: \n',eg1[1:2])
    priceNorm=np.dot(eg1,thetaNormEqn)
    print ('the price via Normal euqation is: \n', priceNorm)

    price=predictiPrice(gradD[0],eg1,features[1],features[2])
    print ('the price via gradient descent is: \n',price)



if __name__=='__main__':
     main()
