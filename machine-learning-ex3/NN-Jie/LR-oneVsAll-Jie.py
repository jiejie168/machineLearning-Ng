__author__ = 'Jie'

'''
Machine learning online class:Exercise 3: one-vs-all.
the files are needed to completed:
lrCostFunction()
oneVsAll()
predictOneVsAll()
predict()
15-03-2019 by Jie Cai
tip: the one column array (n,) and 2D array(n,1) will largely determin the calculation smooth....!!!! important
In the code for fmin_tc, the one column array is used for all the functions and parameters.
'''

import numpy as np
import os, math
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg

class LR_oneVsAll:
    def __init__(self,lambd=0.1,alpha=0.1,num_inters=100):
        self.lambd=lambd
        self.num_inters=num_inters
        self.alpha=alpha

    def loadData(self,fileName):
        '''
        this function is used to read the data types including .m and .txt, which is from Matlab format.
        only the file types ###.## can be used.
        '''
        fileName=str(fileName)
        fileName_list=fileName.split('.') # a list including only two elements

        if fileName_list[1]=="mat":
            data = loadmat('ex3data1.mat')
        else:
            cnames=['Exam1','Exam2','Prediction']  # only suitable for data with 3 columns.
            myData=pd.read_csv(fileName,sep=',',header=None,names=cnames,encoding= 'utf-8')
            store=pd.HDFStore(fileName+".h5")
            store['myData']=myData
            data=store['myData']
            store.close()
            data=data.values  # change the pandas data to numpy array.
        return data

    def displayData(self,X,example_width=None):
        '''
        display 2D data in a nice grid. the data are stored in X.
        it returns the figure handle h and the displayed array if requested.
        :param X:  input dataset (m,n)
        :param example_width:
        :return:
        '''
        # set example_width automatically if not passed in
        if not example_width or not 'example_width' in locals():
            example_width=int(round(math.sqrt(X.shape[1])))

        # closes previously opened figure. preventing a waring after opening too many figures
        plt.close()
        fig=plt.figure()

        # turns 1D X array into 2D
        if X.ndim==1:
            X=np.reshape(X,(-1,X.shape[0]))

        # Gray image
        plt.set_cmap("gray")

        # compute rows, cols
        m,n=X.shape
        example_height=int(n/example_width)


        # compute number of items to display
        display_rows=int(math.floor(math.sqrt(m))) # get the floor of a number
        display_cols=int(math.ceil(m/display_rows))  # get the smallest integer which is larger than argument

        pad=1
        #setup blank display
        display_arrayX=pad+display_rows*(example_height+pad)
        display_arrayY=pad+display_cols*(example_width+pad)
        display_array=-np.ones((int(display_arrayX),int(display_arrayY)))

        # copy each example into a patch on the display array
        curr_ex=1
        for j in range(1,display_rows+1):
            for i in range(1,display_cols+1):
                if curr_ex>m:
                    break
                #copy the path
                #get the max value of the patch to normalize all examples
                max_val=np.max(np.abs(X[curr_ex-1,:]))
                rows=pad+(j-1)*(example_height+pad)+np.array(range(example_height))  # obtain an array
                cols=pad+(i-1)*(example_width+pad)+np.array(range(example_width))    # obtain an array
                display_array[rows[0]:rows[-1]+1,cols[0]:cols[-1]+1]=\
                    np.reshape(X[curr_ex-1,:],(example_height,example_width),order="F")/max_val
                curr_ex=curr_ex+1
            if curr_ex>m:
                break
        # display image
        h=plt.imshow(display_array,vmin=-1,vmax=1)

        # DO NOT show axis
        plt.axis('off')
        #plt.show(block=False)

        return h,display_array

    def _sigmoid(self,z):
        h =1/(1+np.exp(-z))
        return h


    def costFunctionReg(self,theta,X,y):
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
        z=np.dot(X,theta)  # (m,1)
        s=self._sigmoid(z)  #(m,1)
        s[s==1]=0.999  # to avoid the division error. log(1)=0
        predictions=-(1/m)*(y*np.log(s)+(1-y)*np.log(1-s))  # (m,1)
        J=sum(predictions)+(self.lambd/2/m)*sum(theta[1:,0]**2)  # one element arrary
        J=np.squeeze(J)  # costant

        grad_old=np.dot(X.T,(s-y))/m  # (n,1)
        grad_0=grad_old[0]
        grad_1=grad_old[1:] +(self.lambd/m)*theta[1:]
        grad=np.vstack((grad_0,grad_1))
        return J,grad

    def costReg(self,theta,X,y):
        """
        compute cost for logistic regression with regularization
        computes the cost of using theta as the parameter for regularized logistic regression w.r.t. to the parameters.
        :param X: with bias term . (m,n)
        :param y:  (m,)
        :param theta:  (n,)
        :param lambd:
        :return: J
        """
        m=X.shape[0]
        J=0
        #theta=theta.reshape(X.shape[1],1)
        z=np.dot(X,theta)  #(m,)
        #print (z.shape)
        s=self._sigmoid(z)  #(m,)
        #print (z.shape)
        s[s==1]=0.999  # to avoid the division error. log(1)=0
        predictions=-(1/m)*(y*np.log(s)+(1-y)*np.log(1-s))  # (m,)
        J=sum(predictions)+(self.lambd/2/m)*sum(theta[1:]**2)
        J=np.squeeze(J)
        return J

    def gradReg(self,theta,X,y):
        """
        compute gradient for logistic regression with regularization
        computes the gradient of the cost w.r.t. to the parameters.
        :param X: with bias term . (m,n)
        :param y:  (m,1)
        :param theta:  (n,1)
        :param lambd:
        :return: grad
        """
        m=X.shape[0]
        theta=theta.reshape(X.shape[1])
        z=np.dot(X,theta)
        s=self._sigmoid(z) #(m,)
        #s=s.reshape(m,1)
        grad_old=np.dot(X.T,(s-y))/m  # (n,)

        grad_0=np.array([grad_old[0]]) #(1,)
        grad_0=grad_0.reshape((1,1))
        grad_1=grad_old[1:] +(self.lambd/m)*theta[1:] #(n-1,)
        grad_1=grad_1.reshape((theta.shape[0]-1,1))
        grad=np.vstack((grad_0,grad_1)) #(n,)
        grad=np.squeeze(grad)
        return grad

    def gradientDescent(self,theta,X,y):

        m=len(y)
        J_history=[]
        for i in range(self.num_inters):
            cost,grad=self.costFunctionReg(theta,X,y)
            theta=theta-(self.alpha)*grad
            J_history.append(cost)
            #print ("cost in iteration {:d} is {}".format(i,cost))
        J_history=np.squeeze(J_history)
        return  theta, J_history


    def oneVsAll_GD(self,X,y,num_labels):
        '''
        trains multiple logistic regresson classifiers and returns all the classifiers in a matrix
        all_theta, where the i_th row of all_theta corresponding to the classifier for label i
        :param X: X without bias term
        :param y:
        :param num_labels:
        :param lambd:
        :return:
        '''
        m=X.shape[0]
        n=X.shape[1]
        all_theta=np.zeros((num_labels,n+1))
        X=np.c_[np.ones((m,1)),X]

        #set initial theta
        initial_theta=np.zeros((n+1,1))

        for c in range(num_labels):
            # run fmincg to obtain the optima theta
            # this function will return theta and the cost
            # all_theta[c,:]=fmin_cg(self.costFunctionReg,initial_theta,fprime=)
            theta_temp,J_history=self.gradientDescent(initial_theta,X,y)
            all_theta[c,:]=theta_temp.T
        return all_theta

    def oneVsAll_fmin(self,X,y,num_labels):
        '''
        trains multiple logistic regression classifiers and returns all the classifiers in a matrix
        all_theta, where the i_th row of all_theta corresponding to the classifier for label i
        using advanced optimization method: fmin_cg
        '''
        m=X.shape[0]
        all_theta=np.zeros((num_labels,X.shape[1]+1))
        X=np.c_[np.ones((m,1)),X]   # add interception term to X.
        #set initial theta
        initial_theta=np.zeros(X.shape[1])  # (n,)

        for i in range(int(num_labels)):
            label=(y==i).astype(int)
            print ("iteration for the number {}".format(i))
            all_theta[i,:]=fmin_cg(self.costReg,initial_theta,fprime=self.gradReg,args=(X,label),disp=1)
        return all_theta

    def predictOneVsAll(self,all_theta,X,y):
        """
        predict the label for a trained one_vs_all classifier. The labels are in the range 1....K,
        where K=size(all_theta,1)
        :param all_theta:  (num_labels,n+1)
        :param X:  without bias term.  (m,n)
        :return: a vector of predictions for each example in the matrix X. Note that X contains the
        examples in rows. all_theta is a matrix where the i_th row is a trained logistic regression theta
        vector for the i_th class. You should set p to a vector of values from 1..K predicts classes 1,..K.
        """
        m=X.shape[0]
        num_labels=all_theta.shape[0]
        p=np.zeros((X.shape[0],1))
        X=np.c_[np.ones((m,1)),X]

        for i in range(m):
            RX=np.matlib.repmat(X[i,:],num_labels,1)  # (num_labels.n+1)
            RX=RX*all_theta   #(num_labels.n+1)
            SX=np.sum(RX,axis=1)  # ()
            max_val=np.max(SX)
            max_index=np.argmax(SX)
            p[i]=max_index
        p=np.squeeze(p)
        # print (p.shape)
        # print (p)
        # print (y.shape)
        # print (y)
        accuracy=np.mean(p==y)
        return accuracy*100


def main():

    input_layer_size=400  # the number of features. 20*20 input images of digits
    num_labels=10 # 10 labels, from 1 to 10, noted that we have mapped 0 to label 10

    #======part 1 load and visualizing data=======
    #start the exercise by first loading and visualizing the dataset
    # working with a dataset that contains handwriting digits.

    lr_oneVsAll=LR_oneVsAll()
    print ("loading and visualing data.....\n")
    data=lr_oneVsAll.loadData("ex3data1.mat")
    X=data["X"]  # data array  (5000,400)
    y=data["y"]  #(400,1)
    y=np.squeeze(y)   # or y=f.flatten(), change the dimension from (m,1) to 1Dimensional array (m,)
    np.place(y,y==10,0)  # replace the label 10 with 0
    #print (y[:10])
    m=X.shape[0]  # the number of examples
    rand_indices=np.random.randint(0,m,size=100)
    sel=X[rand_indices,:]

    lr_oneVsAll.displayData(sel)
    plt.ion()
    plt.show(block=False)
    plt.pause(0.01)
    print ("Press any key to continue...\n")
    os.system('pause')


    ####=======Part 2a: Vectorize logistic regression ===================
    # in this part, you will reuse your LR code from the last exercise. the task here is
    # to make sure that your regularized logistic regression implementation is vectorized. After
    # that,you will implement one_vs-all classification for the handwritten digit dataset.

    # # test case for lrCostFunction
    # print ("\nTesting lrCostFunction() with regularization")
    # theta_t=np.array([[-2],[-1],[1],[2]])
    # X_t1=np.reshape(np.array(range(1,16)),(5,3),order='F')/10
    # X_t=np.c_[np.ones((5,1)),X_t1]
    # print (X_t)
    # y_t=np.array([[1],[0],[1],[0],[1]])
    # J, grad=lr_oneVsAll.costFunctionReg(theta_t,X_t,y_t)
    # print ("\nCost:{}".format(J))
    # print ("Gradients:\n")
    # print (grad)
    #
    # print ("Program paused. press any key to continue......\n")
    # os.system('pause')

    ###======Part 2b: one_vs-all training============
    print ('\nTraining one-vs-all logistic regression....\n')
    #all_theta=lr_oneVsAll.oneVsAll_GD(X,y,num_labels)
    all_theta=lr_oneVsAll.oneVsAll_fmin(X,y,num_labels)
    print ("the shape of all_theta is {}".format(all_theta.shape))
    #print ("all_theta:{}".format(all_theta[0:2,:]))

    #======Part3: predict for one-vs-all=======
    pred=lr_oneVsAll.predictOneVsAll(all_theta,X,y)
    print ("\nTraining set accuray:{:.2f}".format(float(pred)))


if __name__=="__main__":
    main()










