__author__ = 'Jie'
from pylab import *
'''
this is matlab-like python demo by using pylab.
it is a combinatino of Python, Numpy, Scipy, Matplotlib, and  IPython.
'''


# matrix multiplication
A = rand(3, 3)
A[0:2, 1] = 4
I = A @ inv(A)
I = A.dot(inv(A))
mu=zeros((3,3),'d')
print (mu, mu.shape, mu.shape[0])


# vector manipulations
t = linspace(0, 4, num=1e3)
y1 = cos(t/2) * exp(-t)
y2 = cos(t/2) * exp(-5*t)

# plotting
figure()
plot(t, y1, label='Slow decay')
plot(t, y2, label='Fast decay')
legend(loc='best')
show()

