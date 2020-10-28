"""
Slider that acts as a controller for the plane and streams its values.
"""

import numpy as np
from multiprocessing import Process, Queue, Pipe
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
import math

time = 0.00


def user_input(conn):
    a_min = -math.pi  # the minimial value of the paramater a
    a_max = math.pi  # the maximal value of the paramater a
    a_init = 0  # the value of the parameter a to be used initially, when the graph is created

    b_min = - 50
    b_max = 200
    b_init = 0

    dt = 0.05

    def handle_close(evt):
        conn.close()
        sys.exit()

    def onclick(event):
        global time

        fig.canvas.mpl_disconnect(click)  # disconnects so that second click doesnt raise errors

        conn.send([a_slider.val, b_slider.val, time])  # dont change order, user in inteface on this order
        plt.pause(dt)
        time += dt
        time = round(time, 2)
        onclick('1')  # placeholder

    fig = plt.figure(figsize=(8, 3))
    fig.canvas.mpl_connect('close_event', handle_close)
    click = fig.canvas.mpl_connect('button_press_event', onclick)

    slider_ax1 = plt.axes([0.15, 0.5, 0.75, 0.3])
    slider_ax2 = plt.axes([0.15, 0.1, 0.75, 0.3])

    a_slider = Slider(slider_ax1, 'Elevator Angle', a_min, a_max, valinit=a_init)
    b_slider = Slider(slider_ax2, 'Thrust', b_min, b_max, valinit=b_init)

    plt.plot()
    plt.show()
