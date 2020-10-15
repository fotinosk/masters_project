"""Interface between controller (slider) and model"""

from slider_test import user_input
from multiprocessing import Process, Pipe
import sys


if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=user_input, args=(child_conn,))
    p.start()
    while True:
        try:
            print(parent_conn.recv())
            child_conn.close()
        except:
            print('Shutting down')
            sys.exit()
        # this module tries to close the pipe, but it remains open until
        # the other end of the pipe wants to close too, which only happens
        # when the slider is closed
    p.join()
