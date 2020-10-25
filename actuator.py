"""Implements the actuator class"""


class Actuator:
    """
    Introduces delays to the system
    """

    def __init__(self, tau, dt):
        self.tau = tau  # an array of delays
        self.dt = dt
        self.delay_queue = [[0]*int(i/dt) for i in self.tau]  # a FIFO structure for the delays

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
