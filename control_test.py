"""Module for playing around with the python control library"""

from control.matlab import *
import matplotlib.pyplot as plt

A = [[-0.313, 56.7, 0],
     [-0.0139, -0.426, 0],
     [0, 56.7, 0]]

B = [[0.232],
     [0.0203],
     [0]]

C = [0, 0, 1]

D = [0]

sys = ss(A,B,C,D)

(t,y, x) = bode(sys)
plt.plot(t,y)
plt.show()

