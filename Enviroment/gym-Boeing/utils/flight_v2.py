"""
Redesigned flight class, designed to interanlly handle state initialization and allows
for more options in the control matrices (A,B,C,D)
"""

from utils.model_parameters import Al, Bl, Cl, Dl, dt
from control.matlab import *
import numpy as np
import random


class Flight:
    """
    Simulates the flight.
    Takes in actions and outputs observations.
    """    

    def __init__(self, dt=dt, failure_modes=[[]]):
        """
        Initialize the flight, this includes creating the state space model.
        Allows for the initalization of failure modes, provided that a list of (A,B,C,D)
            matrices is given

        Args:
            dt ([type], optional): Model time increment. Defaults to dt.

            failure_modes (list, optional): A list of failure modes. These modes need to be
            provided as a list of [A',B',C',D']. Multiple can be given in a list of lists
            Constructs the correct mode by default. Defaults to [[]].
        """        
        self.dt = dt
        self.t = 0
        self.sys_norm = StateSpace(Al,Bl,Cl,Dl)

        self.yout = None
        self.last_input = [0,0]
        self.track_out = []

        self.failure_modes = failure_modes

        self.modes = [self.sys_norm]

        # Create failure modes
        if self.failure_modes[0]: # ie the list is not empty
            for mode in self.failure_modes:
                self.modes.append(StateSpace(*mode))

        self.current_mode = random.choice(self.modes)

        if self.current_mode == self.sys_norm:
            self.state = self.state_gen(impulse=True)
        else:
            self.state = self.state_gen()

    def state_gen(self, impulse=False, ds=None):

        """
        Randomly generates the initial state. Allows for impulse to be added in the system
        and for the affected state to be specified for more deliberated evaluation

        Args:
            impulse (bool, optional): Whether or not a system state will be incremented by 10.
            Defaults to False.
            ds (int, optional): The state to be incremented, if not specified, 
            one will be picekd at random. Defaults to None.

        Returns:
            State (list): The current initial state
        """        

        f_vel = (random.random()-0.5) * 0.2         # -0.2 - 0.2
        v_vel = (random.random()-0.5) * 0.4         # -0.2 - 0.2
        p_rate = (random.random()-0.5) * 0.2        # -0.1 - 0.1
        p_angle = (random.random()-0.5) * 0.3       # -0.3 - 0.3
        el_angle = 0
        thrust = 0

        state = [f_vel, v_vel, p_rate, p_angle, el_angle, thrust]

        if impulse:
            if ds is None:
                ds = random.randint(0,3)
            state[ds] += 10
        return state

    def reset(self, ds=None):

        """Reset the flight"""        

        self.t = 0
        self.yout = None
        self.last_input = [0,0]
        self.track_out = []


        '''Must allow for all individual failure modes and all excited states to be called, ideally using the min num of inputs
        It is known that 4 states can be excited, so len of modes+3 (ie 1 failure and 1 norm you get 1+4=5)

        The 1st 4 correspond to random excitation of the normal mode, and the rest to the failure modes
        Start counting from zero
        '''
        possibilities = len(self.modes) + 3


        # Weighted prob selection to ensure that the failure modes come up as often
        # as the normal operation mode
        prob_dist = [0.125] * 4 + (possibilities-4) * [0.5 / (possibilities-4)]

        if ds is None:
            ds = random.choices(np.arange(possibilities), prob_dist)[0]
             

        assert ds < possibilities

        if ds <= 3:
            print('Normal operation mode with state impulse \n')
            self.current_mode = self.sys_norm
            self.state = self.state_gen(impulse=True, ds=ds)
        else:
            print(f"Mode of failre {ds-3} \n")
            self.current_mode = self.modes[possibilities-ds]
            self.state = self.state_gen(impulse=True, ds=ds)

        return ds

    def io(self, inputs):

        """Handles the I/O of the flight sim. Takes in an input and outputs a state

        Returns:
            State (list)
        """        

        yout, _, xout = lsim(self.current_mode, U=[self.last_input, inputs], T=[self.t, self.t+self.dt], X0=self.state)
        self.last_input = inputs[-1]
        self.t += self.dt
        self.state = xout[-1]
        self.yout = yout[-1]
        self.last_input = inputs
        self.track_out.append(self.yout)

        return self.yout

    def show_outputs(self):
        """Show chronological outputs"""
        print(self.track_out)
