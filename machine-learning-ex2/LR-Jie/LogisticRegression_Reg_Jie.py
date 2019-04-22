__author__ = 'Jie'

'''
Logistic regression with regularization
14/03/2019, by Jie. Successful!!

'''

import numpy as np
from scipy.optimize import fmin_tnc
import pandas as pd
import os,sys
import matplotlib.pyplot as plt

class LR_reg:

    def __init__(self,alpha=0.1,num_iters=1000,lambd=1):
        self.alhpa=alpha
        self.num_inters=num_iters
        self.lambd=lambd

    def loadData(self,fileName):
        cnames=['Exam1','Exam2','Prediction']  # only suitable for data with 3 columns.
        myData=pd.read_csv(fileName,sep=',',header=None,names=cnames,encoding= 'utf-8')
        store=pd.HDFStore("myfile.h5")
        store['myData']=myData
        data=store['myData']
        store.close()
        data=data.values  # change the pandas data to numpy array.
        return data

    def plotData(self,X,y):
        '''
        ploting data with + indicating (y=1) examples and o"
        "indicating (y=0) examples
        :param X: the input examples. (m,2), without the bias term
        :param y: the prediction 1, or 0
        :return:
        '''
        pos=[index for index, value in enumerate(y) if value==1]
        neg=[index for index, value in enumerate(y) if value==0]
        plt.plot(X[pos,0],X[pos,1],'k+',linewidth=2,markersize=7,label="y=1")
        plt.plot(X[neg,0],X[neg,1],'ko',linewidth=2,markerfacecolor='yellow',markersize=7,label="y=0")

        plt.xlabel("Microchip Test 1")
        plt.ylabel("Microchip Test 2")
        plt.legend()
        #plt.show()
        return


    def mapFeature(self,X1,X2):
        '''
        feature mapping function to polynomial features
        it maps the two input features to quadratic features used in the regullarization
        exercise.
        returns a new feature array with more features, comprising of X1,2,X1**2,X2**2,
        X1*X2,X1*X2**2,etc.
        input X1,X2 must be the same size
        :param X1:
        :param X2:
        :return:out  # new array with more features
        '''
        degree=6
        m=np.size(X1,axis=0)
        out=np.ones((m,1))

        for i in range(1,degree+1):
            for j in range(i+1):
                n=out.shape[1]
                out=np.insert(out,n,values=(X1**(i-j))*(X2**j),axis=1)
                # print (out)
        return out

    def plotDecisionBoundary(self,theta,X,y):
        '''
        plots the data points X and y into a new figure with the decision bounday defined by
        theta.  plots the data points with + for the positive examples and o for the negative
        examples. X is assumed to be a either
        1) M*3 matrix, where the first column is an all-ones column for the intercept
        2)M*N,N>3 matrix, there the first column is all-ones.
        :param theta:
        :param X:  X is with the bias term
        :param y:
        :return:
        '''
        X=X[:,1:]  # remove the bias term
        self.plotData(X,y)
        if X.shape[1]<=3:
            # only has a single feature. linear
            # only need 2 points to define a line, so choose two endpoints
            # when y=0, x1=min, max, the corresponding x2 values.
            # if self.fit_intercept:
            #     X=self._add_intercept(X)
            plot_x=np.array([np.min(X[:,1])-2,np.max(X[:,1])+2])
            plot_y=-(1/theta[2])*(theta[1]*plot_x+theta[0])
            plt.plot(plot_x,plot_y,label="Decision Bounday")
            plt.legend()
            # plt.axis([30,100,30,100])
            plt.xlim(30,100)
            plt.ylim(30,100)
        else:
            u=np.linspace(-1,1.5,50)
            v=np.linspace(-1,1.5,50)
            z=np.zeros((len(u),len(v)))

            #evaluate z =theta*x over the grid
            for i in range (0,len(u)):
                for j in range (0,len(v)):
                    u_i=np.array([u[i]])
                    v_j=np.array([v[j]])
                    z[i,j]=np.dot(self.mapFeature(u_i,v_j),theta)
            z=z.T


            #fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
            X_u,Y_v=np.meshgrid(u,v)
            plt.contourf(X_u,Y_v,z,1,alpha=.75,cmap=plt.cm.hot)
            contours=plt.contour(X_u,Y_v,z,1,colors="black",linewidth=6)
            #plt.clabel(contours)  # give the description data of contour
        return


    def _sigmoid(self,z):
        h=1/(1+np.exp(-z))
        return  h

    def costFunctionReg(self,X,y,theta):
        """
        compute cost and gradient for logistic regression with regularization
        computes the cost of using theta as the parameter for regularized logistic regression and
        the gradient of the cost w.r.t. to the parameters.
        :param X: with bias term . (m,n)
        :param y:  (m,1)
        :param theta:  (n,1)
        :param lambd:
        :return: J, grad
        """
        m=len(y)
        J=0
        z=np.dot(X,theta)
        s=self._sigmoid(z)
        predictions=-(1/m)*(y*np.log(s)+(1-y)*np.log(1-s))  # (m,1)
        J=sum(predictions)+(self.lambd/2/m)*sum(theta[1:,0]**2)
        J=np.squeeze(J)
        grad_old=np.dot(X.T,(s-y))/m  # (n,1)
        grad_0=grad_old[0]
        grad_1=grad_old[1:] +(self.lambd/m)*theta[1:]
        grad=np.vstack((grad_0,grad_1))
        return J,grad

    def gradientDescent(self,X,y,theta):

        m=len(y)
        J_history=[]

        for i in range(self.num_inters):
            cost,grad=self.costFunctionReg(X,y,theta)
            theta=theta-(self.alhpa)*grad
            J_history.append(cost)
        J_history=np.squeeze(J_history)

        return  theta, J_history

    def predict_prob(self,X,theta):
        # if self.fit_intercept:
        #     X=self._add_intercept(X)
        return self._sigmoid(np.dot(X,theta))

    def accracy(self,X,y,theta,threshold=0.5):
        # X: with bias term
        m=X.shape[0]
        p=np.zeros((m,1))
        prob=self.predict_prob(X,theta)
        for i in range(m):
            if prob[i]>=0.5:
                p[i,0]=1
            else:
                p[i,0]=0
        p=p.flatten()
        p=p.reshape((m,1))
        accuracy=np.mean(p==y)
        return  accuracy*100

    # def gradient(self,X,y,theta,lambd):
    #     m=len(y)
    #     z=np.dot(X,theta)
    #     s=self._sigmoid(z)
    #     grad_old=np.dot(X.T,(s-y))/m  # (n,1)
    #     grad_0=grad_old[0]
    #     grad_1=grad_old[1:] +(lambd/m)*theta[1:]
    #     grad=np.vstack((grad_0,grad_1))
    #
    #     return grad


    # def fit(self, X, y, theta,lambd):   # does not work
    #     opt_weights = fmin_tnc(func=self.costFunctionReg, x0=theta,
    #                   fprime=self.gradient,args=(X, y.flatten()))
    #     return opt_weights[0]


def main():
    ###========load data ===============
    # the first two columns contains the X values and the third column contains the lable (y)
    Lr_reg=LR_reg(alpha=0.1,num_iters=1000)
    data=Lr_reg.loadData("ex2data2.txt")
    X=data[:,0:2]
    y=data[:,2]
    m,n=X.shape
    y=y.reshape((m,1))
    assert(X.shape==(m,2))
    assert (y.shape==(m,1))

    plt.figure(1)
    Lr_reg.plotData(X,y)
    plt.ion()
    plt.show()
    plt.pause(0.01)
    print ("press any key to continue.....\n")
    os.system('pause')

    ##=========Part 1 regularized LR
    # in this part, the given dataset with data points that are not linearly separable.
    #however, we would still like to use LR to classify the data points.
    # to do so, we will introduce more features to use. In particular, we add polynomial
    # features to our data matrix. (similar to polynomial regression)

    #### add polynomial features.

    X=Lr_reg.mapFeature(X[:,0],X[:,1])  # X has already added the column 1.
    print ("the new data set X has the shape of : \n")
    print (X.shape)

    initial_theta=np.zeros((X.shape[1],1))
    lambd=1
    cost,grad=Lr_reg.costFunctionReg(X,y,initial_theta)
    # grad=Lr_reg.gradient(X,y,initial_theta,lambd)
    print ("cost at initial theta (zeros): {}\n".format(cost))
    print ("gradient at initial theta (zeros)-first five values only:\n")
    print (grad[:5])

    print ("Press any key to continue....\n")
    os.system('pause')

    ##======part 2: regularization and accuracies========
    # try different values of lambda and see how regularization affects and decision condart
    # try the following values of lambd(0,1,10,100)
    # lambd=1
    # theta=Lr_reg.fit(X,y,initial_theta,lambd)
    # print (theta)

    theta,J_history=Lr_reg.gradientDescent(X,y,initial_theta)
    print ("the regularized theta using polinomial regression: \n")
    print (theta[:5])
    print ("the regularized J_history using polinomial regression:\n")
    print (J_history[:5])


    # plot boundary
    plt.figure(2)
    Lr_reg.plotDecisionBoundary(theta,X,y)
    plt.ion()
    plt.show()
    plt.pause(10)
    print ("press any key to continue....\n")
    os.system('pause')

    p=Lr_reg.accracy(X,y,theta,0.5)
    print ("train accuracy : {:.1f}\n".format(p))


if __name__=='__main__':
    main()