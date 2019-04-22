__author__ = 'Jie'
'''
Machine Learning class Exercise 3: Part 2:Neural Networks

this file contains code that helps you get started on the linear exercise. YOU
will need to compete the following functions:
lrCostFunction()
oneVsAll()
predictOneVsAll()
predict()
successful completed!!! by Jie 21/03/2019
'''

import numpy as np
import os, math
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt

class NN_prediction:
    def __init__(self,input_layer_size=400,hidden_layer_size=25,num_labels=10):
        self.input_layer_size=input_layer_size
        self.hidden_layer_size=hidden_layer_size
        self.num_labels=num_labels

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

    def predict(self,Theta1,Theta2,X,y):
        '''
        Predict the label of an input given a trained neural network
        outputs the predicted label of X given the trained weights of a neural network (theta1,theta2)
        :param Theta1: (n2,n1)
        :param Theta2:  (num_label,n2+1)
        :param X: (m,n1)
        :param y: (m,1)
        :return:
        '''
        m=X.shape[0]
        num_labels=Theta2[0]
        p=np.zeros((X.shape[0],1))

        X=np.c_[np.ones((m,1)),X]
        n1=X.shape[1] # number of features in the first layer.here is 401
        n2=Theta1.shape[0]  # the number of features in the second layer.without bias here

        z2=np.dot(X,Theta1.T) # (m,n2)
        a2=self._sigmoid(z2)  # (m,n2)

        a2=np.c_[np.ones((a2.shape[0],1)),a2]  #(m,n2+1)
        z3=np.dot(a2,Theta2.T)  #(m,num_label)
        a3=self._sigmoid(z3)  #(m,num_label)

        max_index=np.argmax(a3,axis=1)  #(m,1)
        # print (a3[0])
        # print (max_index[:10])
        max_value=np.max(a3,axis=1)
        # print (max_value[:10])
        ### parameters in theta1,theta2 are based on 1 indexing. so
        ## it is important to add 1 and take the mod w.r.t.10.
        p=(max_index+1)%10  #(m,)
        y=np.squeeze(y) #(m,)
        accuracy=np.mean(p==y)
        return accuracy*100,p

def main():

    #===Part 1: loading and visualizing data=====
    print ("loading and visualizing data.....\n")
    nn_prediction=NN_prediction()
    data=nn_prediction.loadData('ex3data1.mat')
    X=data['X']
    y=data['y']
    m=X.shape[0]
    np.squeeze(y)
    np.place(y,y==10,0)
    # randomly select 100 data points to display
    random_indices=np.random.randint(0,m,size=100)
    sel=X[random_indices,:]

    nn_prediction.displayData(sel)
    plt.ion()
    plt.show()
    plt.pause(0.01)
    print ("program paused. press any key to continue.\n")
    os.system('pause')

    data=nn_prediction.loadData("ex3weights.mat")
    Theta1=data['Theta1']
    Theta2=data['Theta2']
    # print (Theta1.shape)
    # print(Theta2.shape)


    ##=============Part 3: implement predict==============
    accuracy,prob=nn_prediction.predict(Theta1,Theta2,X,y)
    print ("\nTraining set accuracy: {:.2f}".format(accuracy))


    #======run through the examples one at a time to see what it is predicting.
    #randomly permute examples.
    rp=np.arange(m)
    np.random.shuffle(rp)

    for i in range(m):
        #display
        print("\nDisplaying Example image\n")
        tempXi=np.reshape(X[rp[i],:],(1,X.shape[1]))
        nn_prediction.displayData(tempXi)
        plt.ion()
        plt.show()
        plt.pause(0.01)

        accuracy,prob_maxIndex=nn_prediction.predict(Theta1,Theta2,tempXi,y)
        print ("\nNeural Network prediction:{} (digit {})\n".format(prob_maxIndex,prob_maxIndex%10))

        #pause with quit option
        # os.system('pause')
        s=input("Paused-press any key to continue, q to exist:")
        if s=='q':
            break

if __name__=="__main__":
    main()
