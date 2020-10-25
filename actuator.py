"""Implements the actuator class"""


class Actuator:
    """
    Introduces delays to the system
    """

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

    def io(self, u):
        """u is an array of inputs, or an array of input arrays"""
        out = []
        if not isinstance(u[0], list):
            for index in range(len(u)):
                self.delay_queue[index] = [u[index]] + self.delay_queue[index]
                out.append(self.delay_queue[index].pop(-1))
        else:
            for inp in u:
                outp = []
                for index in range(len(inp)):
                    self.delay_queue[index] = [inp[index]] + self.delay_queue[index]
                    outp.append(self.delay_queue[index].pop(-1))
                out.append(outp)

        return out
