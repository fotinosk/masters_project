"""Use this module to declare parameters that will be used throughout the project"""

import numpy as np

dt = 0.05

A = [[-0.0558, 0, -774, 32.2],
     [-0.003865, -0.4342, 0.4136, 0],
     [0.001086, -0.006112, -0.1458, 0],
     [0, 1, 0, 0]]

B = [[0, 5.642],
     [-0.1431, 0.1144],
     [0.003741, -0.4859],
     [0, 0]]

C = [[0, 0, 1, 0],
     [0, 0, 0, 1],
     [-0.003865, -0.4342, 0.4136, 0]]

D = [[0, 0],
     [0, 0],
     [-0.1431, 0.1144]]

A = np.array(A)
B = np.array(B)
C = np.array(C)
D = np.array(D)


elevator_delay = 0.1
thrust_delay = 3.5

input_time_delays = [elevator_delay, thrust_delay]

delay_matrix = np.array([[-1/elevator_delay, 0],
                         [0, -1/thrust_delay]])

# Remake the A,B,C,D matrices to include actuation

A = np.hstack((A, np.zeros((4, 2))))
A_el = np.hstack((np.zeros((2,4)), delay_matrix))
A = np.vstack((A, A_el))

B = np.vstack((B, -1*delay_matrix))

C = np.hstack((C, np.zeros((3, 2))))

D = D

