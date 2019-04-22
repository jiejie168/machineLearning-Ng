__author__ = 'Jie'
'''
Machine learning : Logistic regression
the following functions are needed to completed:
sigmoid(),costFunction(),predict(),costFunctionReg()
12/03/2019-JieCAI (beginning of the year), successful

notes:  the using of gradient descent for optimization has failed.
it should be clarified for the reasons in the future.
'''

import  numpy as np
import sys, os
import  pandas as pd
import  numpy.linalg as alg
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc



class LR:

    def __init__(self,lr=0.01,num_iters=1000,fit_intercept=True,verbose=False):
        self.lr=lr
        self.num_iters=num_iters
        self.fit_intercept=fit_intercept
        self.verbose=verbose

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
        plt.plot(X[pos,0],X[pos,1],'k+',linewidth=2,markersize=7,label="Admitted")
        plt.plot(X[neg,0],X[neg,1],'ko',linewidth=2,markerfacecolor='yellow',markersize=7,label="Not"
                                                                                                "admitted")
        plt.xlabel("Exam 1 score")
        plt.ylabel("Exam 2 score")
        plt.legend()
        #plt.show()
        return


    def plotDecisionBoundary(self,theta,X,y):
        '''
        plots the data points X and y into a new figure with the decision bounday defined by
        theta.  plots the data points with + for the positive examples and o for the negative
        examples. X is assumed to be a either
        1) M*3 matrix, where the first column is an all-ones column for the intercept
        2)M*N,N>3 matrix, there the first column is all-ones.
        :param theta:
        :param X:
        :param y:
        :return:
        '''
        X=X[:,1:]  # remove th
        self.plotData(X,y)
        if X.shape[1]<=3:
            # only has a single feature. linear
            # only need 2 points to define a line, so choose two endpoints
            # when y=0, x1=min, max, the corresponding x2 values.
            if self.fit_intercept:
                X=self._add_intercept(X)
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

            # evaluate z =theta*x over the grid
            for i in range (0,len(u)):
                for j in range (0,len(v)):
                    u_i=np.array([u[i]])
                    v_j=np.array([v[j]])
                    z[i,j]=np.dot(self.mapFeature(u_i,v_j),theta)
            z=z.T
            #fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(10,10))
            X_u,Y_v=np.meshgrid(u,v)
            contours=plt.contour(X_u,Y_v,z,1,colors='black',linewidth=6,label="decision boundary")
            #plt.clabel(contours)  # give the description of contour
            plt.legend()

        return

    def _add_intercept(self,X):
        X=np.c_[np.ones((X.shape[0],1)),X]
        return X

    def _sigmoid(self,z):
        h=1/(1+np.exp(-z))
        return  h

    def _loss(self,X,y,theta):
        if self.fit_intercept:
            X=self._add_intercept(X)
        z=np.dot(X,theta)
        h=self._sigmoid(z)
        loss=(-y*np.log(h)-(1-y)*np.log(1-h)).mean()
        return loss

    # def fit(self,X,y):
    #     if self.fit_intercept:
    #         X=self._add_intercept(X)
    #     theta=np.zeros((X.shape[1],1))
    #     theta_history=np.zeros((self.num_iters+1,X.shape[1]))
    #     loss_history=np.zeros((self.num_iters+1,1))
    #
    #     for i in range(self.num_iters):
    #         z=np.dot(X,theta)
    #         h=self._sigmoid(z)
    #         gradient=np.dot(X.T,(h-y))/y.shape[0]
    #         #print (theta)
    #         #print (gradient)
    #         theta=theta-self.lr*gradient
    #         theta_history[i+1,:]=theta.T
    #         predictions=-y*np.log(h)-(1-y)*np.log(1-h)
    #         loss_history[i+1]=(predictions).mean()
    #
    #         if (self.verbose==True and i % 500==0):
    #             z=np.dot(X,theta)
    #             h=self._sigmoid(z)
    #             #print ('loss:{}'.format(self._loss(X,y,theta)))
    #             #print ('theta:{}'.format((theta)))
    #
    #     return theta,theta_history,loss_history

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


    def cost_function(self, theta, X, y):
        # Computes the cost function for all the training samples
        m = X.shape[0]  # X should add the bias term
        z=np.dot(X,theta)
        h=self._sigmoid(z)
        total_cost=(-y*np.log(h)-(1-y)*np.log(1-h)).mean()
        return total_cost

    def gradient(self, theta, X, y):
        m = X.shape[0]
        z=np.dot(X,theta)
        h=self._sigmoid(z)
        gradient=np.dot(X.T,(h-y))/m
        return gradient

    def fit(self, X, y, theta):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta,
                      fprime=self.gradient,args=(X, y.flatten()))
        return opt_weights[0]




def main():
    ## Load data
    # the first two columns contains the exam scores and the third column contains
    #  the label.
    Lr=LR(lr=0.01,num_iters=1000)
    data=Lr.loadData("ex2data1.txt")
    X=data[:,0:2]
    y=data[:,2]
    m=len(y)
    y=y.reshape((100,1))
    assert (X.shape==(m,2))
    assert (y.shape==(m,1))

    #=========Part1 :plotting=====
    # we start the exercise by first plotting the data to understand the
    # problem we are working with.
    print ("ploting data with + indicating (y=1) examples and o"
           "indicating (y=0) examples.\n")
    plt.figure(1)
    Lr.plotData(X,y)
    plt.ion()
    plt.show()
    plt.pause(0.01)
    print ("\n program paused. press enter to continue.\n")
    os.system('pause')


    ###======Part 2: compute cost and gradient=====
    # in this part, we will implement the cost and gradient for
    # logistic regression.
    m,n=X.shape
    X=np.c_[np.ones((m,1)),X]
    initial_theta=np.zeros(((n+1),1))


    # #optimizing still using gradient descent
    # print ('the theta_history is:\n')
    # print (theta)

    theta = Lr.fit(X, y, initial_theta)
    print ('the theta is:\n')
    print (theta)

    fig=plt.figure(2)
    # lengthX=np.array(list(range(0,len(loss_history))))
    # #print (lengthX)
    # plt.plot(lengthX,loss_history,'b-',linewidth=2,label=r'$\alpha={:.2f}$'.format(0.01))
    Lr.plotDecisionBoundary(theta,X,y)
    plt.ion()
    plt.show()
    plt.pause(0.01)
    os.system('pause')

    #============== Part 4: predict and accuracies
    #############################################
    # after learning the parameters, you will like to use it to predict the outcomes
    # on predict the probability that a student with score 45 on exam 1 and score 85
    # on exam 2 will be addited. Furthermore, compute the training and test set
    # accuracies of our model.
    prob=Lr._sigmoid(np.dot(np.array([1,45,85]),theta))
    print ("for a student with scores 45 and 85,we predict an admission"
           "probability of {:.3f}\n".format(float(prob)))

    ## compute accuracy on our training set
    p=Lr.accracy(X,y,theta,0.5)
    print ("train accuracy:{:.1f}\n".format(p))

if __name__=="__main__":
    main()