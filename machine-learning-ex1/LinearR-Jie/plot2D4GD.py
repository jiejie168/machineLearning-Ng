__author__ = 'Jie'
'''
this code is the code to produce the two-dimentional plot for gradient descent.
the two parameters are theta_0, and theta_1
date : 11/03-2019
'''

import numpy as np
import matplotlib.pyplot as plt

## ### the data to fit
m=20
theta_true=np.array([[2],[0.5]])  # (2,1)
x=np.linspace(-1,1,m)  # (1,m)
x=x.reshape((-1,1))
x=np.c_[np.ones((m,1)),x]   #(m,2)
y=np.dot(x,theta_true)  #(m,1)

fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,6.15))
ax[0].scatter(x[:,1],y,marker='x',s=40,color='k')

def computeCost(X,y,theta):
    '''
    compute the cost for linear regression
    J computes the cost of using theta as the parameter for linear regression
    to fit the data points in X and y.
    :param X: (m,2) ,only for GD with two parameters. n=2
    :param y: (m,1)
    :param theta: (2,1), only for GD with two parameters. n=2
    :return: cost function, J, a scalar.
    '''
    m=len(y) # the number of training examples
    predictions=np.dot(X,theta) # the multiplication of two matrices
    sqrErrors=(predictions-y)**2
    J=1/(2*m)*np.sum(sqrErrors)
    return J

def hypothesis(x,theta):
    return np.dot(x,theta)

def gradientDescent(X,y,theta,alpha,num_iters):
    '''
    to learn theta. thetat updates theta by taking num_iters gradient steps
    with learning rate alpha
    :param X: matrix of X with added bias units
    :param y: Vector of y
    :theta= Vector of theta as np.random.randn(j,1)
    :return: theta, J_history, theta_history
    '''
    m=len(y)
    J_history=np.zeros((num_iters+1,1),'d')
    theta_history=np.zeros((num_iters+1,2))

    for iter in range(0,num_iters):
        prediction=np.dot(X,theta)-y   #  (m,1), X.shape= (m,n), y.shape=(m,1), theta.shape=(n,1)
        temp=np.dot(X.T,prediction)   #(n,1), vectorization! X.T is the derivative of Xi corresponding to each parameter
        theta=theta-alpha*(1/m)*temp # (n,1)
        theta_history[iter+1,:]=theta.T
        J_history[iter+1]=computeCost(X,y,theta)
        #print(theta_history)
    return theta, J_history,theta_history

## first construct a grid of (theta0_value,theta1_value) pairs and obtain their corresponding
##cost function values.
theta0_value=np.linspace(-1,4,101)  # rank one array:  (101,)
theta1_value=np.linspace(-5,5,101)
J_value=np.zeros((len(theta0_value),len(theta1_value)))

for i in range (len(theta0_value)):
    for j in range(len(theta1_value)):
        t=np.array([[theta0_value[i]],[theta1_value[j]]])
        J_value[i,j]=computeCost(x,y,t)

# print (theta0_value.shape)
# print (theta0_value,'\n')
# print (theta1_value,'\n')
# print (J_value,'\n')
# print (J_value.shape)

# a labeled contour plot for the cost function
X,Y=np.meshgrid(theta0_value,theta1_value)
contours=ax[1].contour(X,Y,J_value,30)
ax[1].clabel(contours)
# the target parameter values indicated on the figure of cost function contour
ax[1].scatter(theta_true[0],theta_true[1],s=[50,10],color=['k','w'])

#plt.show()

## take N steps with learning rate alpha down the steepest gradient,
## starting at (theta0,theta1)=(0,0)
N=5
alpha=0.7
theta0=np.zeros((2,1))
theta,J_history,theta_history=gradientDescent(x,y,theta0,alpha,N)
# print (theta,'\n')
# print (J_history,'\n')
#print (theta_history,'\n')

##########annotate the cost function plot with coloured points indicating the
# parameters chosen and red arrows indicating the steps down the gradient.
# also plot the fit function on the plot data.

colors=['b','g','m','c','orange']
ax[0].plot(x[:,1],hypothesis(x,theta0),color=colors[0],lw=2,
           label=r'$\theta_0={:.3f}, \theta_1={:.3f}$'.format(float(theta0[0]),float(theta0[1])))
# plt.ion()
# plt.show()
# plt.pause(0.01)
for i in range(1,N):
    ax[1].annotate('',xy=theta_history[i],xytext=theta_history[i-1],
                   arrowprops={'arrowstyle':'->','color':'r','lw':1},
                   va='center',ha='center')
    ax[0].plot(x[:,1],hypothesis(x,theta_history[i]),color=colors[i],lw=2,
               label=r'$\theta_0={:.3f}, \theta_1={:.3f}$'.format(float(theta_history[i][0]),float(theta_history[i][1])))
    #print (hypothesis(x,theta_history[i]),'\n')

ax[1].scatter(theta0[0]*2,theta0[1]*2,s=[50,10],color=['r','w'])   # the start point on plot
ax[1].scatter(*zip(*theta_history),c=colors,s=40,lw=0)      # the following points on plot

ax[1].set_xlabel(r'$\theta_0$')
ax[1].set_ylabel(r'$\theta_1$')
ax[1].set_title('Cost function')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('data and fit')
axbox=ax[0].get_position()
ax[0].legend(loc=(axbox.x0+0.5*axbox.width, axbox.y0+0.1*axbox.height),fontsize='small')
plt.show()