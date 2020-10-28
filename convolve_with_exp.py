"""Convolve unit pulse with exponential and return truncated result

Assume the pulse lasts 0.05sec, ie the sampling time of the slider
"""

import numpy as np


# dt = 0.05


def exp_convolved_with_step(dt, time_lag, pulse_height, t):
    """Returns the truncated waveform of the convolved function"""
    time_lag = 1 / time_lag

    if t < dt:
        return 1 - np.exp(-time_lag * t)
    else:
        return (1 - np.exp(-time_lag * dt)) * np.exp(-time_lag * (t - dt))


class LaggingActuator:
    """Actuator with lag"""

    def __init__(self, tau, dt, output_dt):
        self.tau = tau  # array
        self.dt = dt
        self.output_dt = output_dt

        self.required_len = int(max(tau) * 2 / output_dt)
        # the length of the time array for the most delayed inputs contribution to decay incremented by output_dt

        self.delay_queue = [[0] * len(self.tau) for _ in range(self.required_len)]

    def __repr__(self):
        return f"Actuator with delays of: {self.tau}, input at increments of: {self.dt}, " \
               f"output at increments of: {self.output_dt}, with the delay queue currently being: {self.delay_queue}"

    def lag_response(self, pulse_height, time_lag):
        """Return a list of the truncated values of the response"""

        # TODO: Implement pulse height

        unit_response = []

        for n in range(self.required_len):
            unit_response.append(exp_convolved_with_step(self.dt, time_lag, pulse_height, t=n*self.output_dt))

        return unit_response

    def io(self, u):
        """Takes input array and outputs output array"""

        # TODO: Overlay the output in the current output queue (sliding rule)

        u0 = self.lag_response(self, pulse_height=u[0], time_lag=self.tau[0])
        print(u0)


a = LaggingActuator([0.1, 3.5], 0.05, 0.01)
print(a.lag_response(1, 3.5))
# a.io([1,0])




