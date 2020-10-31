"""Interface between controller (slider) and model"""

from user_input import user_input
from multiprocessing import Process, Pipe
from state_space_model import model
from initial_state_generator import state
import sys
import warnings
warnings.filterwarnings("ignore")
sys.setrecursionlimit(10**4)

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=user_input, args=(child_conn,))
    p.start()
    input_time = [[], []]
    state = state(danger=False)
    # state = [0, 0, 0, 0, 0, 0]
    while True:
        try:
            received = parent_conn.recv()
            input_time[0].append([received[0], received[1]]), input_time[1].append(received[2])
            child_conn.close()
            if len(input_time[0]) == 20:
                yout, state, time = model(input_time[1], input_time[0], state, plot=True)
                input_time = [[], []]
        except:
            # print(sys.exc_info())  # use when debugging
            print('Shutting down')
            sys.exit()
        # this module tries to close the pipe, but it remains open until
        # the other end of the pipe wants to close too, which only happens
        # when the slider is closed
    p.join()
