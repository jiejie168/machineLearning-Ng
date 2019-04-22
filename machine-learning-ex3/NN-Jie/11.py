__author__ = 'Jie'

import numpy as np
from  scipy.io import loadmat
import sys,math
import matplotlib.pyplot as plt

k_norml=np.linspace(0.58,3,50)
M_norml=1.5-0.5/k_norml**2
plt.figure()
plt.plot(k_norml,M_norml)
plt.show()

