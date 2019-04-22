__author__ = 'Jie'

import numpy as np
from matplotlib import pyplot as plt
import sys, os
import pandas as pd
import numpy.linalg as alg

'''
Machine Learning online course-Exiercise 1: linear regression
this is the self-excise for practice after completed this course after
programming in Matlab.

this file contains code that hepls you get started on the linear exercise. you
will need to complete the following functions in this exericse:

warmUpExercise()
plotData()
gradientDescent()
computeCost()
computeCostMulti()
featureNormalize()
normalEqn()

the main function is listed first in this file.
'''

class linearRegressionmultiVar:
        def load_dataset(self):
            with open('ex1data2.txt','r') as openfile:
                data=openfile.readlines()
                Xdata1,Xdata2,ydata=[],[],[]
                Xdata1=np.array([float(data.split(',')[0]) for data in data])
                Xdata2=np.array([float(data.split(',')[1]) for data in data])
                ydata=np.array([float(data.split(',')[2]) for data in data])
                Xdata=np.c_[Xdata1,Xdata2]
                ydata=ydata.reshape((len(Xdata),1))
                return Xdata, ydata

        def featureNormalize(self,X):
            """
            returns a normalized version of X where the mean value of each feature is
            0 and the standard deviation is 1. this is ofter a good preprocessing step to do when
            working with learning algorithms.
            :param X: the feature data without bias term. (m,n)
            :return: X_norm, mu, sigma:  arrays
            """
            X_norm=X
            n=X.shape[1]
            mu=np.zeros((1,n),'d')
            sigma=np.zeros((1,n),'d')

            for i in range(n):
                mu[0,i]=np.mean(X[:,i],axis=0)  # based on column.
                sigma[0,i]=np.std(X[:,i],ddof=1,axis=0) #  divided by (n-1)
                X_norm[:,i]=(X[:,i]-mu[0,i])/sigma[0,i]

            return  X_norm, mu, sigma

        def computeCostMulti(self,X,y,theta):
            '''
            compute the cost for linear regression
            J computes the cost of using theta as the parameter for linear regression
            to fit the data points in X and y.
            :param X:  the input feature data with bias parameter. (m,n)
            :param y:  the input y data
            :param theta:  parameters  .(n,1)
            :return:  J the cost
            '''
            m=len(y) # the number of training examples
            predictions=np.dot(X,theta) # the multiplication of two matrices
            sqrErrors=(predictions-y)**2
            J=1/(2*m)*np.sum(sqrErrors)
            return  J


        def gradientDescentMulti(self,X,y,theta,alpha,num_iters):
            '''
            to learn theta.  updates theta by taking num_iters gradient steps
            with learning rate alpha
            :param X: matrix of X with added bias units
            :param y: Vector of y
            :theta= Vector of theta as np.random.randn(j,1)
            :return: theta, J_history,theta_history
            '''
            m=len(y)
            J_history=np.zeros((num_iters+1,1),'d')
            theta_history=np.zeros((num_iters+1,3),'d')

            for iter in range(0,num_iters):
                prediction=np.dot(X,theta)-y   #  (m,1), X.shape= (m,n), y.shape=(m,1), theta.shape=(n,1)
                temp=np.dot(X.T,prediction)   #(n,1), vectorization! X.T is the derivative of Xi corresponding to each parameter
                theta=theta-alpha*(1/m)*temp # (n,1)
                theta_history[iter+1,:]=theta.T
                J_history[iter+1]=self.computeCostMulti(X,y,theta)
                #print(theta_history)
            return theta, J_history,theta_history


        def normalEqn(self,X,y):
            XTX=np.dot(X.T,X)
            inv_XTX=alg.inv(XTX)
            inv_XTXXT=np.dot(inv_XTX,X.T)
            theta=np.dot(inv_XTXXT,y)
            return theta

def main():

    ####Initialization
    ###============Part 1: feature normalization========
    print ("Loading data......\n")
    ####Load data
    LRmultiVar=linearRegressionmultiVar()
    X,y=LRmultiVar.load_dataset()
    m=len(y)
    assert (X.shape==(m,2))
    assert(y.shape==(m,1))

    ## print out some data points
    print ("First 10 examples from the dataset: \n")
    print ("x={}\n, y={} \n".format(X[0:5,:],y[0:5]))

    print ("Program paused. Press enter to continue.\n")
    os.system('pause')

    ## scale features and set them to zero mean
    print ("Normalizing Featres....\n")
    X,mu,sigma=LRmultiVar.featureNormalize(X)
    # print (mu)
    # print (sigma)
    # print (X)

    # add bias tearm to X.
    X=np.c_[np.ones((m,1)),X]

    ###========= Part 2: Gradient descent=======
    print ("Running gradient descent.... \n")
    # choose some alpha value
    alpha=0.01
    num_iters=1550

    # init theta and run gradient descent
    theta0=np.zeros((3,1))  # (n,1)
    J=LRmultiVar.computeCostMulti(X,y,theta0)
    theta,J_history,theta_history=LRmultiVar.gradientDescentMulti(X,y,theta0,alpha,num_iters)
    theta1,J_history1,theta_history1=LRmultiVar.gradientDescentMulti(X,y,theta0,0.03,num_iters)
    theta2,J_history2,theta_history2=LRmultiVar.gradientDescentMulti(X,y,theta0,0.1,num_iters)

    # plot the convergence graph
    fig=plt.figure(1)
    lengthX=np.array(list(range(0,len(J_history))))
    #print (lengthX)
    plt.plot(lengthX,J_history,'b-',linewidth=2,label=r'$\alpha={:.2f}$'.format(0.01))
    plt.plot(lengthX,J_history1,'k-',linewidth=2,label=r'$\alpha={:.2f}$'.format(0.03))
    plt.plot(lengthX,J_history2,'r-',linewidth=2,label=r'$\alpha={:.2f}$'.format(0.1))
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.legend(loc='upper right',fontsize='small')
    plt.ion()
    plt.show()
    plt.pause(0.01)
    print ("Program paused. Press enter to continue.\n")
    os.system('pause')

    ### display gradient descent's result
    print ("Theta computed from gradient descent: \n")
    print (theta)
    print ("\n")

    ###===estimate the price of a 1650 sq-ft, 3 bedroom house
    #recall that the first column of X is all-ones, Thus, it does not need to be normalized
    # print (mu.shape)
    # print (sigma.shape)
    area_N=(1650-mu[0,0])/sigma[0,0]
    room_N=(3-mu[0,1])/sigma[0,1]
    predict1=np.dot(np.array([1,area_N,room_N]),theta)
    price=predict1
    print ("predicted price of a 1650 sq-ft, 3 br house"
           "(using gradient descent): \n${:.3f} \n".format(float(price)))
    print ("Program paused. Press enter to continue. \n")
    os.system('pause')

    #####===== Part3 normal equaions=================
    print ("Solving with normal equations...\n")
    # the following code computes the closed form solution for linear regression using
    # the normal equations.

    # load data
    cnames=['area','bdroom','price']
    myData=pd.read_csv('ex1data2.txt',sep=',',header=None,names=cnames,encoding= 'utf-8')
    store=pd.HDFStore('ex1data2.h5')
    store['myData']=myData
    data=store['myData']
    store.close()
    data=data.values  # change the pandas data to numpy array.
    X1=data[:,0:2]
    y1=data[:,2]
    m=len(y1)

    # add intercept term to X
    X1=np.c_[np.ones((m,1)),X1]
    # calculate the parameters from the normal equation
    thetaNorm=LRmultiVar.normalEqn(X1,y1)
    print ("Theta computed from the normal equations: \n")
    print (thetaNorm)
    print ("\n")

    # estimate the price of a 1650 sq-ft, 3 br house
    price=np.dot(np.array([1,1650,3]),thetaNorm)
    print ("predicted price to a 1650 sq-ft,3 br house"
           "(using normal equations): \n ${:.3f}\n".format(float(price)))

if __name__=='__main__':
    main()


















