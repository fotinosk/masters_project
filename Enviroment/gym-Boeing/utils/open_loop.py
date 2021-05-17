from utils.model_parameters import A, B, C, D, dt
from control.matlab import *
import numpy as np
import matplotlib.pyplot as plt

sys_norm = StateSpace(A,B,C,D)

print(bode_plot(sys_norm))