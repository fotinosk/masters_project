"""Interface between controller (slider) and model"""

from user_input import user_input
from multiprocessing import Process, Pipe
from state_space_model import model
import sys


if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=user_input, args=(child_conn,))
    p.start()
    input_time = [[],[]]
    while True:
        # change format to bunches of 20 and to input array, time array
        try:
            received = parent_conn.recv()
            print(received)
            input_time[0].append(received[0]), input_time[1].append(received[1])
            child_conn.close()
            state = None
            if len(input_time[0]) == 20:
                print(input_time)
                yout, state, time = model(input_time[1], input_time[0], state)
                print(yout,state,time)
                input_time = [[], []]
        except:
            print('Shutting down')
            sys.exit()
        # this module tries to close the pipe, but it remains open until
        # the other end of the pipe wants to close too, which only happens
        # when the slider is closed
    p.join()
