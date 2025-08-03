import numpy as np
import pytest
from analysis.gaussian_process import Kernel, MeanProcess, llk, KernelException


# ===========================
# Test the Kernel class
# ===========================

# Define a concrete implementation of Kernel with non-PD kernel to verify exceptions
class NonPDKernel(Kernel):
    def __call__(self, t1, t2):
        return np.abs(np.subtract.outer(t1, t2))


@pytest.fixture
def non_pd_kernel():
    return NonPDKernel()


def test_non_pd_kernel_logdet(non_pd_kernel):
    # Test the log determinant of a non-PD kernel matrix
    t = np.array([0, 1])
    with pytest.raises(KernelException):
        _ = non_pd_kernel.logdet(t, t)


# Define a concrete implementation of Kernel for testing
class TestKernel(Kernel):
    def __call__(self, t1, t2):
        # Simple test kernel implementation: absolute difference and sum
        return np.add.outer(t1, t2) + np.where(np.subtract.outer(t1, t2) == 0, 1, 0)


class TestKernelToeplitz(Kernel):
    def __call__(self, t1, t2):
        # Simple test kernel implementation: absolute difference
        return np.exp(-0.1 * np.abs(np.subtract.outer(t1, t2)))


@pytest.fixture
def test_kernel():
    return TestKernel()


@pytest.fixture
def test_kernel_toeplitz():
    return TestKernelToeplitz()


def test_kernel_call(test_kernel):
    # Test the kernel value between two points
    assert test_kernel(1, 2) == 3
    # Test with arrays
    t1 = np.array([1, 2])
    t2 = np.array([2, 3])
    np.testing.assert_array_equal(test_kernel(t1, t2), np.array([[3, 4], [5, 5]]))


def test_kernel_call_unequal_lengths(test_kernel):
    # Test the kernel with arrays of unequal length
    t1 = np.array([1, 2])
    t2 = np.array([2])
    np.testing.assert_array_equal(test_kernel(t1, t2), np.array([[3], [5]]))


def test_kernel_logdet(test_kernel):
    # Test the log determinant of a 2x2 kernel matrix
    t = np.array([0, 1])
    logdet = test_kernel.logdet(t, t)
    assert np.isclose(logdet, np.log(2))


def test_kernel_apply_inv_right(test_kernel):
    # Test applying the inverse kernel matrix to a vector
    t = np.array([0, 1])
    x = np.array([1, -1])
    np.testing.assert_array_almost_equal(test_kernel.apply_inv_right(t, x),
                                         np.linalg.inv(np.array([[1, 1], [1, 3]])) @ x)


def test_kernel_apply_inv_right_toeplitz(test_kernel_toeplitz):
    # Test applying the inverse kernel matrix to a vector
    t = np.array([0, 1])
    x = np.array([1, -1])
    np.testing.assert_array_almost_equal(test_kernel_toeplitz.apply_inv_right(t, x),
                                         np.linalg.inv(np.array([[1, np.exp(-0.1)], [np.exp(-0.1), 1]])) @ x)


def test_kernel_apply_right(test_kernel):
    # Test applying the kernel matrix to a vector
    t = np.array([0, 1])
    x = np.array([1, -1])
    np.testing.assert_array_almost_equal(test_kernel.apply_right(t, x), np.array([[1, 1], [1, 3]]) @ x)


def test_kernel_non_integer_inputs(test_kernel):
    # Test the kernel with non-integer inputs
    assert test_kernel(0.5, 1.5) == 2.0
    np.testing.assert_array_almost_equal(test_kernel(np.array([0.5, 1.5]), np.array([1.5, 2.5])),
                                         np.array([[2.0, 3.0], [4.0, 4.0]]))


def test_kernel_negative_inputs(test_kernel):
    # Test the kernel with negative time points
    assert test_kernel(-2, -1) == -3
    np.testing.assert_array_equal(test_kernel(np.array([-2, -1]), np.array([-1, 0])), np.array([[-3, -2], [-1, -1]]))


def test_kernel_large_arrays(test_kernel):
    # Test with larger arrays of time points
    t1 = np.linspace(0, 10, 5)
    t2 = np.linspace(5, 15, 5)
    expected = np.add.outer(t1, t2) + np.where(np.subtract.outer(t1, t2) == 0, 1, 0)
    np.testing.assert_array_equal(test_kernel(t1, t2), expected)


def test_kernel_non_equidistant_spacing(test_kernel):
    # Test with non-equidistant spacing between time points
    t1 = np.array([0, 1, 3])
    t2 = np.array([2, 4])
    expected = np.add.outer(t1, t2) + np.where(np.subtract.outer(t1, t2) == 0, 1, 0)
    np.testing.assert_array_equal(test_kernel(t1, t2), expected)


def test_kernel_edge_cases(test_kernel):
    # Test kernel with edge cases: very large numbers
    assert test_kernel(1e9, 1e9 + 1) == 2e9 + 1


# ===========================
# Test the MeanProcess class
# ===========================

# Define a concrete implementation of MeanProcess for testing
class TestMeanProcess(MeanProcess):
    def __init__(self, a=1, b=0):
        self.a = a
        self.b = b

    def __call__(self, t, **kwargs):
        a = kwargs.get('a', self.a)
        b = kwargs.get('b', self.b)
        return a * t + b


@pytest.fixture
def test_mean_process():
    return TestMeanProcess()


def test_mean_process_call(test_mean_process):
    # Test the mean value at a single time point
    assert test_mean_process(5) == 5
    # Test with parameters
    assert test_mean_process(5, a=2, b=1) == 11


# ===========================
# Test the llk function
# ===========================

class DiagonalPSDKernel(Kernel):

    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, t1, t2):
        dt = np.subtract.outer(t1, t2)
        return np.where(dt == 0, self.sigma ** 2, 0)


@pytest.fixture
def diagonal_kernel():
    return DiagonalPSDKernel(sigma=2)


def test_llk_perfect_fit(diagonal_kernel, test_mean_process):
    kernel = diagonal_kernel
    # Test single point
    t = np.array([0])
    Y = test_mean_process(t)
    mu = test_mean_process(t)
    assert llk(t, Y, mu, kernel) == -0.5 * len(t) * np.log(2 * np.pi) - 0.5 * np.log(kernel.sigma ** 2)

    # Test 2x2 matrix
    t = np.array([0, 1])
    Y = test_mean_process(t)
    mu = test_mean_process(t)
    assert llk(t, Y, mu, kernel) == -0.5 * len(t) * np.log(2 * np.pi) - 0.5 * np.log(kernel.sigma ** 2) * 2


def test_llk_noisy(diagonal_kernel, test_mean_process):
    kernel = diagonal_kernel
    # Test single point
    t = np.array([0])
    Y = test_mean_process(t) + 1
    mu = test_mean_process(t)
    non_overlap_llk = -0.5 * len(t) * np.log(2 * np.pi) - 0.5 * np.log(kernel.sigma ** 2)
    assert llk(t, Y, mu, kernel) - non_overlap_llk == -0.5 * 1 / kernel.sigma ** 2

    # Test 2x2 matrix
    t = np.array([0, 1])
    Y = test_mean_process(t) + 1
    mu = test_mean_process(t)
    non_overlap_llk = -0.5 * len(t) * np.log(2 * np.pi) - 0.5 * np.log(kernel.sigma ** 2) * 2
    assert llk(t, Y, mu, kernel) - non_overlap_llk == -0.5 * 2 / kernel.sigma ** 2


def test_llk_edge_cases(diagonal_kernel):
    # Mismatched lengths should raise ValueError
    with pytest.raises(ValueError):
        llk(np.array([0, 1]), np.array([0]), np.array([0]), diagonal_kernel)
    with pytest.raises(ValueError):
        llk(np.array([0]), np.array([0, 1]), np.array([0]), diagonal_kernel)
    with pytest.raises(ValueError):
        llk(np.array([0]), np.array([0]), np.array([0, 1]), diagonal_kernel)

    # Empty arrays should raise ValueError due to mismatch in dimensions
    with pytest.raises(ValueError):
        llk(np.array([]), np.array([]), np.array([]), diagonal_kernel)

    # 2D arrays should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        llk(np.array([0]), np.array([[0]]), np.array([[0]]), diagonal_kernel)
