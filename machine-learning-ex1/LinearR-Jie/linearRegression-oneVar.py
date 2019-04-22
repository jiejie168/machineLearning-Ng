__author__ = 'Jie'
# Python 3

import numpy as np
import sys,os
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

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

x refers to the population size in 10,000s
y reers to the profit in $10,000s
'''

class LinearRegressionOneVar:


    def warmUpExercise(self):
        A=np.eye(5)
        print (A)

    def load_dataset(self):
        with open('ex1data1.txt','r') as openfile:
            data=openfile.readlines()
            Xdata,ydata=[],[]
            Xdata=np.array([float(data.split(',')[0]) for data in data])
            ydata=np.array([float(data.split(',')[1]) for data in data])
            Xdata=Xdata.reshape((len(Xdata),1))
            ydata=ydata.reshape((len(Xdata),1))
            return Xdata, ydata

    def plotData(self,X,y):
        '''
        plots the data points and gives the figure axes labels of population
         and profit
        :param X:
        :param y:
        :return:
        '''

        #plt.figure(1)
        plt.plot(X,y,'rx',markersize=10)
        ax=plt.gca()
        plt.xlabel("Profit in $ 10,000s")
        plt.ylabel("Population of City in 10,000s")
        # plt.ion()
        # plt.show()
        # plt.pause(0.01)

    def computeCost(self,X,y,theta):
        '''
        compute the cost for linear regression
        J computes the cost of using theta as the parameter for linear regression
        to fit the data points in X and y.
        :param X:
        :param y:
        :param theta:
        :return:
        '''
        m=len(y) # the number of training examples
        predictions=np.dot(X,theta) # the multiplication of two matrices
        sqrErrors=(predictions-y)**2
        J=1/(2*m)*np.sum(sqrErrors)
        return J


    def gradientDescent(self,X,y,theta,alpha,num_iters):
        '''
        to learn theta. thetat updates theta by taking num_iters gradient steps
        with learning rate alpha
        :param X: matrix of X with added bias units
        :param y: Vector of y
        :theta= Vector of theta as np.random.randn(j,1)
        :return: theta, J_history
        '''
        m=len(y)
        J_history=np.zeros((num_iters+1,1),'d')
        theta_history=np.zeros((num_iters+1,2))

        for iter in range(0,num_iters):
            prediction=np.dot(X,theta)-y   #  (m,1), X.shape= (m,n), y.shape=(m,1), theta.shape=(n,1)
            temp=np.dot(X.T,prediction)   #(n,1), vectorization! X.T is the derivative of Xi corresponding to each parameter
            theta=theta-alpha*(1/m)*temp # (n,1)
            theta_history[iter+1,:]=theta.T
            J_history[iter+1]=self.computeCost(X,y,theta)
            #print(theta_history)
        return theta, J_history,theta_history

def main():
    print ('Running warmUpExercise........\n')
    print ('5*5 Identity Matrix:  \n')
    LROnvVar=LinearRegressionOneVar()
    LROnvVar.warmUpExercise()

    print ('Program paused. Press enter to continue. \n')
    os.system('pause')

    # part 2 :plotting###======================================================
    print ('ploting data......\n')
    X1,y=LROnvVar.load_dataset()
    m=len(y)  # the number of examples
    assert (X1.shape==(m,1))
    assert(y.shape==(m,1))

    plt.figure(1)
    LROnvVar.plotData(X1,y)
    plt.ion()
    plt.show()
    plt.pause(0.1)
    print("Program paused. Press enter to continue. \n")
    os.system('pause')

    #######part3 cost and gradient descent==========================
    X=np.c_[np.ones((m,1)),X1]  # add a column of ones to X
    theta=np.zeros((2,1)) # initialize fitting parameters

    # some gradient descent settings
    iterations=1500
    alpha=0.01
    print("\nTesting the cost function.....\n")
    # compute and display initial cost
    J=LROnvVar.computeCost(X,y,theta)
    print ("with theta=[0:0] \nCost computed = {:5.2f}".format(J))

    ## further testing of the cost function
    J=LROnvVar.computeCost(X,y,np.array([[-1],[2]]))
    print ("with theta=[-1:2] \nCost computed = {:5.2f}".format(J))
    print ("Program paused. press enter to continue.\n")
    os.system('pause')
    plt.pause(0.01)

    # run gradient descent
    theta1,J_history,theta_history=LROnvVar.gradientDescent(X,y,theta,alpha,iterations)
    print ("Theta found by gradient descent:\n")
    print (theta1)

    ### plot the linear fit
    #plt.figure(2)
    plt.plot(X[:,1],np.dot(X,theta1),'k-')
    #plt.legend('Training data')
    #plt.legend(loc='lower right')
    plt.ion()
    plt.show()
    plt.pause(0.1)
    print ("Program paused. press enter to continue.\n")
    os.system('pause')

    ###### predict values for population sizes of 35000 and 70000
    predict1=np.dot(np.array([1, 3.5]),theta1)
    predict2=np.dot(np.array([1,7]),theta1)
    print (predict1)
    print ("For populartion =35000, we predict a profit of {:10.3f}\n".format(float(predict1)*10000))
    print ("For populartion =35000, we predict a profit of {:10.3f}\n".format(float(predict2)*10000))

    print("Program paused. press enter to continue.\n")
    os.system('pause')

    ### Part 4: visualizing J (theta_0,thet_1)=========================
    print ("Visualizing  J (theta_0,thet_1) .....\n")

    # Grid over which we will calculate J
    theta0_vals=np.linspace(-10,10,100)  # rank one array
    theta1_vals=np.linspace(-1,4,100)

    # initialize J_vals to a matrix of 0's
    J_vals=np.zeros((len(theta0_vals),len(theta1_vals))) # (100,100) array

    # fill out J_vals
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t=np.array([[theta0_vals[i]],[theta1_vals[j]]])
            J_vals[i,j]=LROnvVar.computeCost(X,y,t)

    J_vals=J_vals.T
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
    theta0_vals,theta1_vals=np.meshgrid(theta0_vals,theta1_vals)
    contours=ax.contour(theta0_vals,theta1_vals,J_vals,np.logspace(-2,3,20))
    ax.scatter(theta1[0],theta1[1],s=[50,10],color=['k','w'])
    ax.clabel(contours)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    plt.ion()
    plt.show()
    plt.pause(0.1)
    print ("Program paused. press enter to continue.\n")
    os.system('pause')

    fig=plt.figure(4)
    ax=fig.gca(projection='3d')
    cset=ax.contour(theta0_vals,theta1_vals,J_vals,50)
    plt.plot(theta1[0],theta1[1],'rx',markersize=10,linewidth=2)
    plt.ion()
    plt.show()
    plt.pause(0.1)
    print ("Program paused. press enter to continue.\n")
    os.system('pause')

if __name__=="__main__":
    main()













