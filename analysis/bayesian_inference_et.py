from typing import Union

import numpy as np
from scipy.special import erf
from numpy.typing import NDArray

from analysis.gaussian_process import Kernel, MeanProcess


class PosControlKernel(Kernel):

    def __init__(self, K: float, sig_r: float, sig_y: float, gamma: float):
        self._K = K
        self._sig_r = sig_r
        self._sig_y = sig_y
        self._gamma = gamma

    def __call__(self, t1: Union[float, NDArray], t2: Union[float, NDArray]) -> Union[float, NDArray]:
        return cov_mat(t1, t2, self._K, self._sig_r, self._sig_y, self._gamma)


class PosControlMeanProcess(MeanProcess):

    def __init__(self, K: float):
        self._K = K

    def __call__(self, t: Union[float, NDArray], **kwargs) -> Union[float, NDArray]:
        """
        Calculate the mean of the process at time t.
        :param t: time(s)
        :param kwargs:
        :return: the mean of the process at time t.
        """
        if 'v' not in kwargs:
            raise ValueError('"v" must be provided as a keyword argument.')
        v = kwargs['v']
        x_0 = 0
        return x_0 * np.exp(-self._K * t) + v * (t + np.exp(-self._K * t) / self._K - 1 / self._K)


class VelControlKernel(Kernel):

    def __init__(self, K: float, b: float, sig_r: float, sig_y: float):
        self._K = K
        self._b = b
        self._sig_r = sig_r
        self._sig_y = sig_y

    def __call__(self, t1: Union[float, NDArray], t2: Union[float, NDArray]) -> Union[float, NDArray]:
        t1 = np.asarray(t1)
        t2 = np.asarray(t2)
        t1 = t1[..., np.newaxis]
        t2 = t2[np.newaxis, ...]

        dt = t1 - t2
        min_t = np.where(t1 < t2, t1, t2)

        K_eff = self._K + self._b

        diag = np.where(dt == 0, self._sig_y ** 2, 0)
        prefactor = self._sig_r ** 2 / (2 * K_eff ** 3)
        exp_diff = np.exp(-K_eff * np.abs(dt))
        exp_sum = np.exp(-K_eff * (t1 + t2))
        exp_t1 = np.exp(-K_eff * t1)
        exp_t2 = np.exp(-K_eff * t2)
        return diag + prefactor * (2 * K_eff * min_t - 2 + 2 * exp_t1 + 2 * exp_t2 - exp_sum - exp_diff)


class VelControlMeanProcess(MeanProcess):

    def __init__(self, K: float, b: float):
        self._K = K
        self._b = b

    def __call__(self, t: Union[float, NDArray], **kwargs) -> Union[float, NDArray]:
        """
            Calculate the mean of the process at time t.
            :param t: time(s)
            :param kwargs:
            :return: the mean of the process at time t.
            """
        if 'v' not in kwargs:
            raise ValueError('"v" must be provided as a keyword argument.')
        v = kwargs['v']
        K_eff = self._K + self._b
        v_0 = 0
        return 1 / K_eff * (v_0 - self._K / K_eff * v) * (1 - np.exp(-K_eff * t)) + self._K / K_eff * v * t


class VelVelControlKernel(Kernel):

    def __init__(self, K: float, b: float, sig_r: float, sig_y: float, large_t=False):
        """
        Create a kernel for the *velocity* of the velocity control process.

        :param K: control parameter
        :param b: bias parameter
        :param sig_r: standard deviation of the process noise at zero lag
        :param sig_y: standard deviation of the observation noise
        :param large_t: If True, use the approximation for large t for faster computation.
        """
        self._K = K
        self._b = b
        self._sig_r = sig_r
        self._sig_y = sig_y
        self._K_eff = K + b
        self._large_t = large_t

    def __call__(self, t1: Union[float, NDArray], t2: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Calculate the covariance *of the velocity* between two time points t1 and t2.

        :param t1: A single time point or an array of time points.
        :param t2: A single time point or an array of time points.
        :return: the covariance between t1 and t2
        """
        t1 = np.asarray(t1)
        t2 = np.asarray(t2)
        t1 = t1[..., np.newaxis]
        t2 = t2[np.newaxis, ...]
        dt = t1 - t2
        n = len(t1)

        diag = np.diag(np.ones(n), 0)
        tridiag = 2*diag + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
        meas_term = self._sig_y ** 2 * tridiag

        prefactor = self._sig_r ** 2 / (2 * self._K_eff)
        exp_diff = np.exp(-self._K_eff * np.abs(dt))
        exp_sum = np.exp(-self._K_eff * (t1 + t2)) if not self._large_t else 0
        proc_noise_term = prefactor * (exp_diff - exp_sum)
        return meas_term + proc_noise_term


class VelVelControlMeanProcess(MeanProcess):

    def __init__(self, K: float, b: float, large_t=False):
        """
        Create a mean process for the *velocity* of the velocity control process.
        :param K: control parameter
        :param b: bias parameter
        :param large_t: If True, use the approximation for large t for faster computation.
        """
        self._K = K
        self._b = b
        self._K_eff = K + b
        self._large_t = large_t

    def __call__(self, t: Union[float, NDArray], **kwargs) -> Union[float, NDArray]:
        """
        Calculate the mean of the *velocity* process at time t.
        :param t: time(s)
        :param kwargs: Additional keyword arguments.
        :return: the mean of the process at time t.
        """
        if 'v' not in kwargs:
            raise ValueError('"v" must be provided as a keyword argument.')
        v = kwargs['v']
        v_0 = 0
        bias_factor = self._K / self._K_eff
        if self._large_t:
            return np.ones_like(t) * bias_factor * v
        else:
            return v_0 * np.exp(-self._K_eff * t) + (1 - np.exp(-self._K_eff * t)) * bias_factor * v


def cov_mat(t1: Union[int, float, NDArray], t2: Union[int, float, NDArray],
            K: float, sig_r: float, sig_y: float, gamma: float) -> float:
    """
    Calculate the covariance between two time points t1 and t2.
    :param t1: first time point
    :param t2: second time point
    :param K: control parameter
    :param sig_r: standard deviation of the process noise at zero lag
    :param sig_y: standard deviation of the observation noise
    :param gamma: correlation time of the process noise
    :return: the covariance between t1 and t2
    """
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    t1 = t1[..., np.newaxis]
    t2 = t2[np.newaxis, ...]
    dt = t1 - t2
    T = t1 + t2
    diag = np.where(dt == 0, sig_y ** 2, 0)
    prefactor = sig_r ** 2 / (2 * K)
    if gamma == 0:
        process_cov = np.exp(-K * np.abs(dt)) - np.exp(-K * T)
    else:
        def half(x, y):
            arg1 = (x + K * gamma ** 2) / (np.sqrt(2) * gamma)
            arg2 = (x - y + K * gamma ** 2) / (np.sqrt(2) * gamma)
            arg3 = K * gamma / np.sqrt(2)
            arg4 = (x - K * gamma ** 2) / (np.sqrt(2) * gamma)
            A = erf(arg1) * np.exp(K * (x - y))
            B = erf(arg2) * np.exp(K * (x - y))
            C = erf(arg3) * np.exp(-K * (x + y))
            D = erf(arg4) * np.exp(-K * (x + y))
            return A - B - C - D

        term = half(t1, t2) + half(t2, t1)
        with np.errstate(divide='ignore'):  # Suppress log(0) warnings
            if np.any(np.isnan(np.log(term))):
                raise ValueError(f'Error at K={K}, sig_r={sig_r}, sig_y={sig_y}, gamma={gamma}, '
                                 f'term={term}, log(term)={np.log(term)}')
            ln = (K ** 2 * gamma ** 2) / 2 + np.log(term)
        process_cov = np.exp(ln)
    return diag + prefactor * process_cov
