"""Module for playing around with the python control library"""

from control.matlab import *
import matplotlib.pyplot as plt

A = [[0.00187, 0.0263, -86.15, -31.939],
     [-0.07, -0.2941, 672.9, -4.12],
     [0.002653, -0.0009834, -0.3449, 0.0003729],
     [0, 0, 1, 0]]

B = [[1.93],
     [-15.21],
     [-0.9686],
     [0]]

C = [[0, 0, 0, 1]]

D = [0]

sys = ss(A,B,C,D)

(t,y) = step(sys)
plt.plot(y,t)
plt.show()

