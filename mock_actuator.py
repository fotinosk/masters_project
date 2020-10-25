"""Mock actuator"""

from control.matlab import *
from state_space_model import sys
import matlab.engine

eng = matlab.engine.start_matlab()

A = [0]
B = [0, 0]
C = [[0], [0]]
D = [[2, 0], [0, 2]]

H = StateSpace(A, B, C, D)

tau = [0.1, 3.5]

print(lsim(H, U=[[1, 1], [2, 2], [3, 3], [4, 4]], T=[1, 2, 3, 4]))

sys2 = eng.delayss(A, B, C, D, tau)

eng.quit()
