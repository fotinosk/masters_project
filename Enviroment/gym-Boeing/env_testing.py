from utils.model_parameters import A, B, C, D,dt
from control.matlab import * 
import numpy as np
import matplotlib.pyplot as plt

sys = StateSpace(A,B,C,D)

x = 20

state0 = [0,0,0,0,0,0]
state1 = [x,0,0,0,0,0] # increase in forward vel
state2 = [0,4*x,0,0,0,0] # increase in vertical vel
state3 = [0,0,4*x,0,0,0] # increase in pitch rate
state4 = [0,0,0,x,0,0] # increase in pitch angle

action = [0,0]
actions = [[0,0]] * 10000

t = np.arange(0,0.05*10000, 0.05)

yout, _, xout = lsim(sys, U=actions , T=t , X0=state2)

err = [np.linalg.norm(x, 1) for x in yout]

plt.plot(err)
plt.show()
