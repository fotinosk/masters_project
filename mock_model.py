"""
Mock airplane to model to begin the inmplementation of the reinforcment learning model.
Plane steady-space model is taken from the 3F1 module experiment:
F4E Model 3, Altitude: 35000ft, Mach: 0.9
"""

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import control

# 1 input 1 output, change later

a_f4e_3 = [[0, 0, -1, 0],
           [0, -0.667, 18.11, 84.34],
           [0, 0.08201, -0.6587, -10.81],
           [0, 0, 0, -14]]

b_f4e_3 = [[0], [-85.1], [0], [14]]

c_f4e_3 = [1, 0, 0, 0]

d_f4e_3 = [0]

mock = signal.lti(a_f4e_3, b_f4e_3, c_f4e_3, d_f4e_3)

# t0, y0 = mock.step()
# y = mock.output([1, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5])
#
# plt.plot(y[0], y[1])
# plt.plot(t0, y0)
# plt.show()
