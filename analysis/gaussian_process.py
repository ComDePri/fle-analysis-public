"""
Define Gaussian process abstract classes and functions.
"""

from abc import ABC, abstractmethod
from typing import Union
import warnings

import numpy as np
import xarray as xr
from numpy.typing import NDArray
import scipy as scp
from xarray_einstats.stats import logsumexp
from util.FLE_utils import is_toeplitz


class KernelException(Exception):
    """Exception for kernel errors."""
    pass


class Kernel(ABC):
    """Base class for kernels in the Bayesian analysis."""

    @abstractmethod
    def __call__(self, t1: Union[float, NDArray], t2: Union[float, NDArray]) -> Union[float, NDArray]:
        """Return the kernel value between time steps t1 and t2.
        :param t1: The first time point, or a vector of time points.
        :param t2: The second time point, or a vector of time points.
        :return: The kernel value between times t1 and t2.
        """
        pass

    def logdet(self, t1: Union[float, NDArray], t2: Union[float, NDArray]) -> float:
        """
        Return the log determinant of the kernel matrix between times t1 and t2.
        Note: If the Kernel is not positive definite, a KernelException is raised.
        :param t1: The first time point, or a vector of time points.
        :param t2: The second time point, or a vector of time points.
        :return: The log determinant of the kernel matrix between times t1 and t2.
        """
        mat = self(t1, t2)
        if len(t1) == 1 and len(t2) == 1:
            return np.log(mat)
        try:
            L = scp.linalg.cholesky(mat)
        except scp.linalg.LinAlgError as e:
            warnings.warn('Cholesky decomposition failed. Attempting full logdet calculation.')
            sgn, logdet = np.linalg.slogdet(mat)
            if sgn <= 0:
                raise KernelException('Kernel matrix is not positive definite.') from e
            return logdet
        logdet = 2 * np.sum(np.log(np.diag(L)))
        return logdet

    def apply_inv_right(self, t: Union[float, NDArray], x: Union[float, NDArray]) -> NDArray:
        """Return the inverse kernel matrix evaluated at t X t applied to x.
        :param t: The time point, or a vector of time points.
        :param x: The vector to apply the inverse kernel matrix to, taken at time(s) t.
        :return: The inverse kernel matrix evaluated at t X t applied to x.
        """
        mat = self(t, t)
        if len(t) == 1:
            return 1 / mat * x
        if is_toeplitz(mat):
            res = scp.linalg.solve_toeplitz(mat[0, :], x)
        else:
            res = scp.linalg.solve(mat, x, assume_a='pos')
        return res

    def apply_right(self, t: Union[float, NDArray], x: Union[float, NDArray]) -> NDArray:
        """Return the kernel matrix between evaluated at t X t applied to x.
        :param t: The time point, or a vector of time points.
        :param x: The vector to apply the kernel matrix to, taken at time(s) t.
        :return: The kernel matrix between evaluated at t X t applied to x.
        """
        mat = self(t, t)
        if len(t) == 1:
            mat = [[mat]]
        return mat @ x


class MeanProcess(ABC):
    """Base class for mean processes in the Bayesian analysis."""

    @abstractmethod
    def __call__(self, t: int, **kwargs) -> float:
        """Return the mean of the process at time t.
        :param t: The time point.
        :param kwargs: Additional keyword arguments - external time varying inputs, known (non-inferrable) parameters,
        etc.
        :return: The mean of the process at time t.
        """
        pass


def llk(t: Union[float, NDArray], Y: Union[float, NDArray], mu: Union[float, NDArray], kernel: Kernel) -> float:
    """Return the log likelihood of the data given the mean process and kernel.
    :param t: The time points.
    :param Y: The data at the corresponding time points.
    :param mu: The mean process.
    :param kernel: The kernel of the process.
    :return: The log likelihood of the data given the mean process and kernel.
    """
    t = np.asarray(t)
    Y = np.asarray(Y)
    mu = np.asarray(mu)
    if len(t) != len(Y) or len(t) != len(mu):
        raise ValueError(f'Time points and vector lengths do not match: {len(t)}, {len(Y)}, {len(mu)}.')
    if Y.shape != mu.shape:
        raise ValueError(f'Data and mean process shapes do not match: {Y.shape}, {mu.shape}.')
    if len(Y.shape) != 1:
        raise NotImplementedError(f'Data and mean process must be 1D arrays.')
    if len(t) == 0:
        raise ValueError('Empty time points.')
    if len(Y) == 0:
        raise ValueError('Empty data.')
    if len(mu) == 0:
        raise ValueError('Empty mean process.')

    n = len(Y)
    logdet = kernel.logdet(t, t)
    Y_res = Y - mu
    return -0.5 * (n * np.log(2 * np.pi) + logdet + Y_res @ kernel.apply_inv_right(t, Y_res))


def marginalize_lse(x: type(xr.DataArray), dims) -> type(xr.DataArray):
    """
    Marginalize a log-sum-exp array along the specified dimensions.
    :param x: an xarray DataArray containing log values
    :param dims: dimensions to reduce
    :return: the marginalized log-sum-exp array
    """
    return logsumexp(x, dims)


def uniform_prior(params: dict) -> type(xr.DataArray):
    """
    Create a uniform prior over the parameters.
    :param params: dictionary of parameter names and values. Values should be lists of possible values.
    :return: the prior
    """
    shape = [len(v) for v in params.values()]
    prior_vals = np.ones(shape) / np.prod(shape)
    prior = xr.DataArray(np.log(prior_vals), coords=params, dims=params.keys())
    return prior
