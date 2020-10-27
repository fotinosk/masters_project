"""Reformatting Actuator class to include lag instead of delay"""


class LagActuator:
    """Introduces lag in the system inputs"""

    def __init__(self, tau, dt, initial_inputs=None):
        self.tau = tau  # an array of delays
        self.dt = dt

        if initial_inputs is None:
            self.initial_inputs = [0] * len(self.tau)
        else:
            self.initial_inputs = initial_inputs

        # a FIFO structure for the delays
        self.delay_queue = [[initial_inputs[i]]*int(self.tau[i]/dt) for i in range(len(self.tau))]

    def __repr__(self):
        return str(self.delay_queue)

    def io(self,u):
        """u is an array of inputs, or an array of input arrays"""

    def trunc_expon(self, delay_time):
        """returns the values of the truncated exponential function"""
