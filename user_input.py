"""
Slider that acts as a controller for the plane and streams its values.
"""

import numpy as np
from multiprocessing import Process,Queue,Pipe
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys

time = 0.00


def user_input(conn):
    a_min = -100  # the minimial value of the paramater a
    a_max = 100  # the maximal value of the paramater a
    a_init = 0  # the value of the parameter a to be used initially, when the graph is created

    dt = 0.05

    def handle_close(evt):
        conn.close()
        sys.exit()

    def onclick(event):
        global time

        # print(time, a_slider.val)
        conn.send([a_slider.val, time])

        plt.pause(dt)
        time += dt
        time = round(time, 2)
        onclick('1')  # placeholder

    fig = plt.figure(figsize=(8, 3))
    fig.canvas.mpl_connect('close_event', handle_close)
    fig.canvas.mpl_connect('button_press_event', onclick)

    slider_ax = plt.axes([0.1, 0.25, 0.8, 0.5])

    a_slider = Slider(slider_ax, 'Slider', a_min, a_max, valinit=a_init)

    plt.plot()
    plt.show()



