__author__ = 'Jie'
'''
machine larning onlince class. Exercise 4: Neural Network learning
the folloing files are needed to completed:
sigmoidGradient()
randInitializeWeights()
nnCostFunction()

'''

import  numpy as np
from scipy.optimize import fmin_cg
import os,sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math
import numpy.linalg as lg

class NN_learning:

    def __init__(self,input_layer_size=400,hidden_layer_size=25,num_labels=10,lambd=1):

        self.input_layer_size=input_layer_size
        self.hidden_layer_size=hidden_layer_size
        self.num_labels=num_labels
        self.lambd=lambd

    def loadData(self,fileName):
        '''
        this function is used to read the data types including .m and .txt, which is from Matlab format.
        only the file types ###.## can be used.
        '''
        fileName=str(fileName)
        fileName_list=fileName.split('.') # a list including only two elements

        if fileName_list[1]=="mat":
            data = loadmat(fileName)
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

    def _sigmoidGradient(self,z):
        '''
        returns the gradient of the sigmoid function, evaluated at z
        this should work regardless if z is a matrix or a vector.
        :param z:
        :return:
        '''
        g=np.zeros(z.shape)  # g has the same shape with z.
        g=self._sigmoid(z)*(1-self._sigmoid(z))
        return g

    def nnCostFunction(self,nn_params,X,y):
        '''
        implements the neural network cost function for a two layer neural network which performs
        classification. The returned parameters grad should be a "unrolled" vector of the partial
        derivatives of the neural network
        :param nn_params:
        :param X:  without bais term (m,input_layer_size)
        :param y:  (m,1)
        :param lambd:
        :return:
        '''
        ##===reshape nn_parameters back into the parameters Theta1 and Theta2, the
        #weights matrices for our 2 layer neural network.
        n1=self.input_layer_size+1  #401
        n2=self.hidden_layer_size   # 25
        n3=self.num_labels   #10
        J=0

        theta1=np.reshape(nn_params[:n2*(n1)],(n2,n1))
        theta2=np.reshape(nn_params[n2*(n1):],(n3,n2+1))

        m=X.shape[0]
        X=np.c_[np.ones((m,1)),X]

        ##====compute the overall cost

        z2=np.dot(X,theta1.T) # (m,n1)*(n1,n2)=(m,n2)
        a2=self._sigmoid(z2) # (m,n2)

        a2=np.c_[np.ones((a2.shape[0],1)),a2]  #(m,n2+1)
        z3=np.dot(a2,theta2.T)  # (m,n2+1)*(n2+1,n3)=(m,n3)
        a3=self._sigmoid(z3)    #(m,n3)

        y_1=np.zeros((m,n3))  # the n3 class outputs  (m,n3)
        for i in range(m):
            y_1[i,y[i]-1]=1    #  noted that the original data theta1 and theta2 are based on matlab, so the index is starring from 1 instead of 0. A minus 1 is needed.
        eps=1e-15
        loss=-y_1*np.log(a3)-(1-y_1)*np.log(1-a3+eps)  #(m,n3)
        J=np.sum(np.sum(loss,axis=1))/m

        # the following code is via the non_vectarization method.
        # for i in range(m):
        #     z2=np.dot(X[i,:],theta1.T) # (1,n1)*(n1,n2)=(1,n2)
        #     a2=self._sigmoid(z2) # (1,n2)
        #     a2=a2.reshape((1,n2))
        #
        #     a2=np.c_[np.ones((1,1)),a2] #(1,n2+1)
        #     z3=np.dot(a2,theta2.T)  # (1,n2+1)*(n2+1,n3)=(1,n3)
        #     a3=self._sigmoid(z3)    #(1,n3)
        #
        #     yi=np.zeros((1,n3))
        #     yi[0,y[i]-1]=1
        #     loss=-yi*np.log(a3)-(1-yi)*np.log(1-a3)  #(1,n3)
        #     J=J+np.sum(loss)
        # J=J/m

        Jreg=(self.lambd/2/m)*(np.sum(np.sum(theta1[:,1:]**2))+np.sum(np.sum(theta2[:,1:]**2)))
        J=J+Jreg

        ####===obtain the gradient via BP.

        theta1_grad=np.zeros(theta1.shape)  # (n2,n1)
        theta2_grad=np.zeros((theta2.shape))  #(n3,n2+1)

        # # ======using non-vectorization loop.
        # delta1_sum=np.zeros(theta1.shape)  # (n2,n1)
        # delta2_sum=np.zeros((theta2.shape))  #(n3,n2+1)
        #
        # for i in range(m):
        #     delta3=a3[i,:]-y_1[i]   #(1,n3)
        #     delta3=delta3.reshape((1,n3))
        #     z2_temp=np.concatenate((np.ones(1,),z2[i,:]))  #(1,n2+1)
        #     z2_temp=z2_temp.reshape((1,n2+1))
        #     delta2=np.dot(delta3,theta2)*self._sigmoidGradient(z2_temp)  #(1,n2+1)
        #     delta2=delta2.flatten() # (,n2)
        #     delta2=delta2[1:].reshape((1,n2))
        #
        #     Xi=X[i,:].reshape((1,n1))
        #     a2i=a2[i,:].reshape((1,n2+1))
        #     delta1_sum=delta1_sum+np.dot(delta2.T,Xi)  # (n2,n1)
        #     delta2_sum=delta2_sum+np.dot(delta3.T,a2i) # (n3,n2+1)
        # theta1_grad=delta1_sum/m
        # theta2_grad=delta2_sum/m

        # ##=======using vectorization
        delta3=a3-y_1  # (m,n3), (m-n3)=(m-n3)
        delta3=delta3.reshape((m,n3))
        z2=z2.reshape((m,n2))
        g=self._sigmoidGradient(z2)  #(m,n2)
        #delta2=np.dot(delta3,theta2[:,1:])*g  # (m,n3), (n3,n2)=(m,n2)
        delta2=np.multiply(np.dot(delta3,theta2)[:,1:],g)

        theta1_grad=np.dot(X.T,delta2)  # (n1,m), (m,n2)
        theta2_grad=np.dot(a2.T,delta3)   # (n2+1,m),(m,n3)

        theta1_grad=theta1_grad.T/m
        theta2_grad=theta2_grad.T/m
        ###########################=================

        grad=np.concatenate((theta1_grad.flatten(),theta2_grad.flatten()))
        #print (grad.shape)
        return  J, grad

    def randInitializeWeights(self,L_in,L_out):
        '''
        randomly initialize the weights of a layer with L_in incoming connections and
        L_out outgoing connections
        Note that W should be set to a matrix of size (L_out,1+L_in) as the first
        column of W handles the bias terms
        :param L_in:
        :param L_out:
        :return:  W. (L_out,1+L_in)
        '''
        epsilon_init=0.12
        W=np.random.rand(L_out,1+L_in)*2*epsilon_init-epsilon_init  # range btw (-epsilon, epsilon)
        return W

    def debugInitialieWeights(self,fan_out,fan_in):
        '''
        initialize the weights of a layer with fan_in incoming connectins and fan_out outfoing
        connections using a fixed strategy, this will help you later in debugging
        :param fan_out:
        :param fan_in:
        :return: the weight W
        '''
        W=np.zeros((fan_out,1+fan_in))
        num=np.sin(np.arange(W.size))/10
        W=num.reshape(W.shape)
        return W

    def computeNumericalGradient(self,J,theta):
        '''
        compute the gradient using "finite differences and gives us a numerical estimation of
        the gradient.Calling y=J(theta) should return the function value at theta.
        notes: the folloiwng code implements numerical gradient checking, and returns the
        numerical gradient. It sets numgrad(i) to ( a numerical approximation of ) the partial
        derivative of J w.r.t.the i-th input argument, evalated at theta.
        :param theta:
        :return:
        '''
        numgrad=np.zeros(theta.shape)
        perturb=np.zeros(theta.shape)
        e=10**(-4)

        for p in range(theta.size):
            # set perturbation vector
            perturb[p]=e
            loss1,_=J(theta-perturb)
            loss2,_=J(theta+perturb)
            # print ("\n",np.c_[loss1,loss2])
            numgrad[p]=(loss2-loss1)/2/e
            # print ("\n",numgrad)
            perturb[p]=0

        return numgrad

    def checkNNGrandients(self):
        '''
        creat a small neural network to check the backpropagation gradients. it will output the
        analytical gradients produced bu the bp code and the numerical gradients (using computeNumerical
        Gradient). these two gradietns computations should result in very similar values.
        :param lambd: the paramter for regularization.
        :return:
        '''
        # generate some 'random' test data
        theta1=self.debugInitialieWeights(self.hidden_layer_size,self.input_layer_size)
        theta2=self.debugInitialieWeights(self.num_labels,self.hidden_layer_size)

        # reusing degugInitializeWeights to generate X
        m=5
        X=self.debugInitialieWeights(m,self.input_layer_size-1)
        y=np.mod(np.arange(m),self.num_labels)
        y=1+y.T

        # unroll parameters.
        nn_params=np.concatenate((theta1.flatten(),theta2.flatten()))
        # numGrad=self.computeNumericalGradient(nn_params,X,y)  # (nn_params.shape)
        # _,grad=self.nnCostFunction(nn_params,X,y)  #(nn_params,shape)
        #print (grad)
        #print (numGrad)

        # Short hand for cost function
        costFunc=lambda p:self.nnCostFunction(p,X,y)
        (cost,grad)=costFunc(nn_params)
        numGrad=self.computeNumericalGradient(costFunc,nn_params)

        #visually examine the two gradient computations. The two columns you get
        # should be very similar.
        print (np.c_[numGrad,grad])

        # evaluate the norm of the difference btw two solutions.
        # if you have a correct implementation, and assuming you used EPSILON=0.0001,
        # in computeNumerialGradient, then diff below should be less than 1e-9

        diff=lg.norm(numGrad-grad)/lg.norm(numGrad+grad)
        print ("if your bp implementation is correct, then \n....."
               "the relative difference will be small (less than 1e-9).\n"
               "\nRelative difference: {}\n ".format(diff))


def main():
    #===========loading and visualizing data
    nn_learning=NN_learning()
    print ("Loading and visualizing data....\n")
    data1=nn_learning.loadData("ex4data1.mat")
    data2=nn_learning.loadData("ex4weights.mat")
    X=data1["X"]
    y=data1["y"]
    np.place(y,y==10,0)
    y=np.squeeze(y)
    theta1=data2["Theta1"]
    theta2=data2["Theta2"]
    # print (theta1.shape)
    # print (theta2.shape)

    ## randomly select 100 data points to display
    m=X.shape[0]
    sel=np.random.randint(0,m,size=100)
    sel=X[sel,:]
    nn_learning.displayData(sel)
    plt.ion()
    plt.show()
    plt.pause(0.01)
    os.system("pause")

    ##=======unroll parameters=====================================
    nn_params=np.concatenate((theta1.flatten(),theta2.flatten()))
    print (nn_params.shape)   # (10285,), one dimensional array

    ##=======part3: compute cost (Feedforward)
    # first implement the feeforward cost without regularization.

    print("\nFeedforward using neural networ.....\n")
    J,_=nn_learning.nnCostFunction(nn_params,X,y)
    print ("cost at parameters(loaded from ex4weights):{:.5f}".format(J))

    ##=====Part five:sigmoid gradient=============
    print ("\nEvaluating sigmoid gradient...\n")
    g=nn_learning._sigmoidGradient(np.array([-1,-0.5,0,0.5,1]))
    print ("sigmoid gradient evaluated at [-1,-0.5,0,0.5,1]: \n")
    print (g)
    print ("\n\n")
    os.system("pause")

    #========Part 6, initializing parameters===============
    # in this part, you will be starting to implement a two layer nn that
    # classifies digits. You will start by implementing a function to initilize
    # the weights of the nn.
    print ("\nInitializing Neural Network parameters....\n")
    initial_Theta1=nn_learning.randInitializeWeights(nn_learning.input_layer_size,nn_learning.hidden_layer_size)
    initial_Theta2=nn_learning.randInitializeWeights(nn_learning.hidden_layer_size,nn_learning.num_labels)

    # unroll parameters
    initial_nn_params=np.concatenate((initial_Theta1.flatten(),initial_Theta2.flatten()))

    #===Part 7 implement backpropagation=============
    print ("\nChecking Backpropagation....\n")
    _,grad=nn_learning.nnCostFunction(nn_params,X,y)
    # print (grad)

    nn_learning=NN_learning(input_layer_size=3,hidden_layer_size=5,num_labels=3,lambd=0)
    nn_learning.checkNNGrandients()
    os.system("pause")



if __name__ == '__main__':
    main()






























