"""
Module for user controlled input to the model.
User constantly inputs data using slider and output is plotted on graph.
"""

import control
from scipy import signal


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

sys = control.StateSpace(A, B, C, D)

tf = control.ss2tf(sys)  # get transfer function

# print(tf.issiso())  # is single input single output

iosys = control.LinearIOSystem(sys)
# print(iosys.dcgain())

sign = signal.lti(A, B, C, D)
print(sign.output([10, 10, 10, 30, 300, 10, 20, 50], [1, 2, 3, 4, 5, 6, 7, 8]))
# inputs and timestamps and outputs the output
