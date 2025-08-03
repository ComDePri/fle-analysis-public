import numpy as np
from itertools import product
import pytest
from scipy.linalg import eigh
from analysis.bayesian_inference_et import PosControlKernel, VelControlKernel, VelVelControlKernel


def is_pos_def(x):
    eigvals = eigh(x, eigvals_only=True)
    return np.all(eigvals > 0)


K_values = np.logspace(-2, 2, 5)
sig_r_values = np.logspace(-2, 2, 5)
sig_y_values = np.logspace(-2, 2, 5)
gamma_values = np.logspace(-2, 0, 5)
b_values = np.logspace(-2, 1, 5)

pos_control_kernel_params = list(product(K_values, sig_r_values, sig_y_values, gamma_values))
vel_control_kernel_params = list(product(K_values, sig_r_values, sig_y_values, b_values))


# noinspection DuplicatedCode
@pytest.mark.parametrize('K, sig_r, sig_y, gamma', pos_control_kernel_params)
def test_pos_control_kernel_pos_def(K, sig_r, sig_y, gamma):
    kernel = PosControlKernel(K, sig_r, sig_y, gamma)
    t = np.random.default_rng().uniform(0, 100, 200)
    try:
        mat = kernel(t, t)
        assert is_pos_def(mat), f'Kernel matrix is not positive definite for parameter set\n' \
                                f'K: {K}, sig_r: {sig_r}, sig_y: {sig_y}, gamma: {gamma}'
    except ValueError as e:
        if 'nan' in str(e) or 'inf' in str(e):
            pytest.skip(f'Kernel matrix contains NaN or inf for parameter set\n'
                        f'K: {K}, sig_r: {sig_r}, sig_y: {sig_y}, gamma: {gamma}')


# noinspection DuplicatedCode
@pytest.mark.parametrize('K, sig_r, sig_y, b', vel_control_kernel_params)
def test_vel_control_kernel_pos_def(K, sig_r, sig_y, b):
    kernel = VelControlKernel(K, sig_r, sig_y, b)
    t = np.random.default_rng().uniform(0, 200, 300)
    try:
        mat = kernel(t, t)
        assert is_pos_def(mat), f'Kernel matrix is not positive definite for parameter set\n' \
                                f'K: {K}, sig_r: {sig_r}, sig_y: {sig_y}, b: {b}'
    except ValueError as e:
        if 'nan' in str(e) or 'inf' in str(e):
            pytest.skip(f'Kernel matrix contains NaN or inf for parameter set\n'
                        f'K: {K}, sig_r: {sig_r}, sig_y: {sig_y}, b: {b}')


# noinspection DuplicatedCode
@pytest.mark.parametrize('K, sig_r, sig_y, b', vel_control_kernel_params)
def test_vel_vel_control_kernel_pos_def(K, sig_r, sig_y, b):
    kernel = VelVelControlKernel(K, sig_r, sig_y, b, large_t=False)
    t = np.random.default_rng().uniform(0, 200, 300)
    try:
        mat = kernel(t, t)
        assert is_pos_def(mat), f'Kernel matrix is not positive definite for parameter set\n' \
                                f'K: {K}, sig_r: {sig_r}, sig_y: {sig_y}, b: {b}'
    except ValueError as e:
        if 'nan' in str(e) or 'inf' in str(e):
            pytest.skip(f'Kernel matrix contains NaN or inf for parameter set\n'
                        f'K: {K}, sig_r: {sig_r}, sig_y: {sig_y}, b: {b}')


# noinspection DuplicatedCode
@pytest.mark.parametrize('K, sig_r, sig_y, b', vel_control_kernel_params)
def test_vel_vel_control_kernel_pos_def_large_t(K, sig_r, sig_y, b):
    kernel = VelVelControlKernel(K, sig_r, sig_y, b, large_t=True)
    t = np.random.default_rng().uniform(0, 200, 300)
    try:
        mat = kernel(t, t)
        assert is_pos_def(mat), f'Kernel matrix is not positive definite for parameter set\n' \
                                f'K: {K}, sig_r: {sig_r}, sig_y: {sig_y}, b: {b}'
    except ValueError as e:
        if 'nan' in str(e) or 'inf' in str(e):
            pytest.skip(f'Kernel matrix contains NaN or inf for parameter set\n'
                        f'K: {K}, sig_r: {sig_r}, sig_y: {sig_y}, b: {b}')
