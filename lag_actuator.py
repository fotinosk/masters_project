"""Convolve unit pulse with exponential and return truncated result

For some reason the output is scaled by 5
"""

import numpy as np
from operator import add


def exp_convolved_with_step(dt, time_lag, pulse_height, t):
    """Returns the truncated waveform of the convolved function"""
    time_lag = 1 / time_lag

    if t < dt:
        return 1 - np.exp(-time_lag * t)
    else:
        return (1 - np.exp(-time_lag * dt)) * np.exp(-time_lag * (t - dt))


class LaggingActuator:
    """Actuator with lag

    Note: The dt has to be equal to output_dt otherwise there will be more outputs per unit time that inupts,
    causing the system to destabilize
    """

    def __init__(self, tau, dt, output_dt):
        self.tau = tau  # array
        self.dt = dt
        self.output_dt = output_dt

        self.required_len = int(max(tau) * 2 / output_dt)
        # the length of the time array for the most delayed inputs contribution to decay incremented by output_dt

        self.lag_queue = [[0] * self.required_len for _ in range(len(self.tau))]

    def __repr__(self):
        return f"Actuator with delays of: {self.tau}, input at increments of: {self.dt}, " \
               f"output at increments of: {self.output_dt}, with the delay queue currently being: {self.lag_queue}"

    def lag_response(self, pulse_height, time_lag):
        """Return a list of the truncated values of the response"""

        unit_response = []

        for n in range(self.required_len):
            unit_response.append(round(exp_convolved_with_step(self.dt, time_lag, pulse_height, t=n*self.output_dt), 4))

        unit_response = [pulse_height*sample_value for sample_value in unit_response]

        return unit_response

    def io(self, u):
        """Takes input array and outputs output array with time increment of output_dt"""

        output = []

        if isinstance(u[0], list):
            # ie multiple inputs as a list of lists => [[a,b],[c,d],...]
            for inp in u:
                outp = []
                for index in range(len(inp)):
                    response = self.lag_response(pulse_height=inp[index], time_lag=self.tau[index])
                    response = response[::-1]
                    self.lag_queue[index] = list(map(add, self.lag_queue[index], response))
                    self.lag_queue[index] = [0] + self.lag_queue[index]
                    outp.append(self.lag_queue[index].pop(-1))
                output.append(outp)
            return output

        for i in range(len(u)):
            response = self.lag_response(pulse_height=u[i], time_lag=self.tau[i])
            response = response[::-1]  # invert the response, since the 1dt result is the one that comes out 1st
            # so the delay queue remains FIFO

            self.lag_queue[i] = list(map(add, self.lag_queue[i], response))
            self.lag_queue[i] = [0] + self.lag_queue[i]  # sliding rule
            output.append(self.lag_queue[i].pop(-1))

        return output

# Uncomment for demonstration purposes

"""
a = LaggingActuator([0.1, 3.5], 0.05, 0.05)
x = [[0, 0] for i in range(499)]
x = [[10, 10]] + x

import matplotlib.pyplot as plt
t = np.linspace(0, 25, num=500)

y = a.io(x)
y0 = [i[0] for i in y]
y1 = [i[1] for i in y]

plt.plot(t, y0)
plt.plot(t, y1)
plt.show()
"""
