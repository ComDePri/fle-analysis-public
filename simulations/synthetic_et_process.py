from ComDePy.numerical import Agent
from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Union


class PropControlAgentException(Exception):
    """Base class for exceptions in this module."""
    pass


class PropControlAgent(Agent):
    """Base class for agents in the proportional control simulation."""
    _agent_counter = 0  # Counter for the number of agents. Used to initialize the random number generator.

    def __init__(self, K, x0, target: Union[NDArray, Callable],
                 step_noise: Union[NDArray, Callable, None],
                 meas_noise: Union[NDArray, Callable, None], max_steps: int):
        """

        :param K: The control gain of the process.
        :param x0: The initial state of the process.
        :param target: The target state of the process. Can be a callable which takes a time step as input and returns a
        target value, or a 1D array of target values.
        :param step_noise: The noise in the process. Can be a callable which takes a random number generator, a time
        step, and the current state as input and returns a noise value, or a 1D array of noise values. If None, no noise
        is added.
        :param meas_noise: The noise in the measurements. Can be a callable which takes a random number generator, a
        time step, and the current state as input and returns a noise value, or a 1D array of noise values. If None, no
        noise is added.
        :param max_steps: The number of time steps to simulate.
        """

        # Make sure the input parameters are valid.
        if K < 0:
            raise PropControlAgentException('K must be non-negative.')
        if max_steps <= 1:
            raise PropControlAgentException('max_steps must be greater than 1.')

        self._K = K
        self._x0 = x0
        self._max_steps = max_steps

        self._t = np.arange(max_steps)  # Time steps
        self._x = np.empty(max_steps)  # State of the process
        self._x_meas = np.empty(max_steps)  # Measurements of the process
        self._x[0] = x0  # Initialize the state of the process

        # Make sure the target, step_noise, and meas_noise parameters are valid.
        if isinstance(target, np.ndarray):
            if target.ndim != 1:
                raise PropControlAgentException('target must be a 1D array.')
            self._target = lambda t: target[t]
        elif callable(target):
            self._target = target
        else:
            raise PropControlAgentException('target must be a callable or a 1D array.')

        if isinstance(step_noise, np.ndarray):
            if step_noise.ndim != 1:
                raise PropControlAgentException('step_noise must be a 1D array.')
            self._step_noise = lambda rng, t, x: step_noise[t]
        elif callable(step_noise):
            self._step_noise = step_noise
        elif step_noise is None:
            self._step_noise = lambda rng, t, x: 0
        else:
            raise PropControlAgentException('step_noise must be a callable or a 1D array.')

        if isinstance(meas_noise, np.ndarray):
            if meas_noise.ndim != 1:
                raise PropControlAgentException('meas_noise must be a 1D array.')
            self._meas_noise = lambda rng, t, x: meas_noise[t]
        elif callable(meas_noise):
            self._meas_noise = meas_noise
        elif meas_noise is None:
            self._meas_noise = lambda rng, t, x: 0
        else:
            raise PropControlAgentException('meas_noise must be a callable or a 1D array.')

    def _sim(self):
        """Simulate the process step by step."""
        for t in self._t[1:]:
            self._x[t] = self._step(self._x[t - 1], t - 1)
        for t in self._t:
            self._x_meas[t] = self._x[t] + self._meas_noise(self._rng, t, self._x[t])

    def _step(self, x, t):
        """Calculate the next state of the process."""
        return x + self._dx(x, t)

    @abstractmethod
    def _dx(self, x, t):
        """Calculate the change in the state of the process."""
        pass

    def _initialize_seed(self) -> None:
        """Initialize the random number generator."""

        # Initialize the random number generator using the agent counter.
        self._rng = np.random.default_rng(PropControlAgent._agent_counter)
        PropControlAgent._agent_counter += 1


class PosControlAgent(PropControlAgent):
    """
    Simulate a process controlled by proportional control with a target position.
    """

    def __init__(self, K, x0, target, step_noise, meas_noise, max_steps):
        super().__init__(K=K, x0=x0, target=target, step_noise=step_noise,
                         meas_noise=meas_noise, max_steps=max_steps)

    def _dx(self, x, t):
        """
        Calculate the change in the state of the process according to a discrete proportional control mechanism.
        :param x: current state of the process.
        :param t: current time step.
        :return: the change in the state of the process.
        """
        return -self._K * (x - self._target(t)) + self._step_noise(self._rng, t, x)

    def _sim(self):
        """Simulate the process."""
        super()._sim()

        # Save the state of the process and the measurements into public attributes.
        self.t = np.arange(self._max_steps)
        self.x = self._x
        self.x_meas = self._x_meas


class VelControlAgent(PropControlAgent):
    """Simulate a process controlled by proportional control with a target velocity.
    An interface over the PropControlAgent class, with clearer semantics for the velocity control use case."""

    def __init__(self, K, b, v0, target, step_noise, meas_noise, max_steps):
        super().__init__(K=K, x0=v0, target=target, step_noise=step_noise,
                         meas_noise=meas_noise, max_steps=max_steps)
        self._b = b

    def _dx(self, v, t):
        """Calculate the change in the state of the process according to a discrete proportional control mechanism."""
        return -self._K * (v - self._target(t)) - self._b * v + self._step_noise(self._rng, t, v)

    def _sim(self):
        """Simulate the process."""
        super()._sim()

        # Save the state of the process and the measurements into public attributes.
        self.t = np.arange(self._max_steps)
        self.v = self._x
        # The measured variable is the position, so the measurement noise is added to the position.
        noises = np.array([self._meas_noise(self._rng, t, v) for t, v in enumerate(self.v)])
        self.x_meas = np.cumsum(self.v) + noises
