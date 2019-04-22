__author__ = 'Jie'
 # coding=utf8

from pylab import *
import numpy as np
from matplotlib import pyplot as pp
import math as mm
import numpy.linalg  as alg
import scipy.io


'''
def displayData(X):


    #set example_width automatically if not passed in
    example_width=round(np.sqrt(X.shape[1]))
    #pp.colormap(gray)
    m,n=X.shape[0],X.shape[1]
    example_height=(n/example_width)

    # compute number of items to display
    display_rows=np.floor(np.sqrt(m))
    display_cols=np.ceil(m/display_rows)

    pad=1
    # setup blank  display
    display_array=-np.ones((pad+display_rows*(example_height+pad),pad+display_cols*(example_width+pad)),'d')

    curr_ex=1
    for j in range(0,display_rows+1):
        for i in range(0,display_cols+1):
            if curr_ex>m:
                break

            max_val=np.max(abs(X[curr_ex,:]))
            display_array[pad+(j-1)*(example_height+pad)+(1:example_height),\
                          pad+(i-1)*(example_width+pad)+(1:example_width)]=

'''

def sigmoid(z):
    # compute sigmoid function
    #j=sigmoid(z) compute the sigmoid of z.
    g=1.0/(1.0+exp(-z))
    return g


def lrGostFunction(theta,X,y,lambda1):
    m=len(y)
    J=0
    grad=zeros(size(theta,0),'d')

    z=X.dot(theta)  #theta is a vector, (n+1)*1
    g=sigmoid(z)
    predictions=-y*log(g)-(1-y)*log(1-g)

    J=(1/m)*sum(predictions)+(lambda1/2/m)*sum(theta[1:]*theta[1:])

    grad=(1/m)*(X.T).dot(g-y)
    temp=theta
    temp[0]=0
    grad=grad+(lambda1/m)*temp
    return J, grad

#def oneVsAll(X,y,num_labels,lambda1):



#def  predictOneVsAll(all_theta,X):


def main():

    print ('loading and visulizing data.....\n')
    mat=scipy.io.loadmat('ex3data1') # A dictionary
    X=mat['X']  # X is a 5000*400 array.
    m=X.shape[0] # number of sample

    input_layer_size=400  # 20*20 input images of digits
    num_labels=10   # 10 labels, from 1 to 10.

    rand_indices=np.random.randint(0,m,size=100)
    sel=X[rand_indices,:]

    # the test case for lrCostFunction
    theta_t=array([[-2],[-1],[1],[2]])
    one=ones((5,1),'d')
    X_t=linspace(1,15,15)/10
    X_t=X_t.reshape(5,3,order='F')
    X_t=c_[one,X_t]
    y_t=array([[1],[0],[1],[0],[1]])
    lambda_t=3
    print ('\n Testing  lrCostFunction() with regularization \n')
    J,grad=lrGostFunction(theta_t,X_t,y_t,lambda_t)

    print ('\n Cost: %f\n'%J)



if __name__=='__main__':
    main()