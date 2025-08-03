import pytest
import numpy as np
from numpy import nan
from numpy.testing import assert_array_almost_equal

from util.FLE_utils import align_series, embed, two_point_correlation, is_toeplitz


# ========================================
# ===== Test the align_series function ===
# ========================================


def test_align_series_equal_length():
    t1 = np.array([1, 2])
    t2 = np.array([0, 1])
    idx = [0, 1]
    series = [t1, t2]
    aligned_series, zero_idx = align_series(series, idx)
    np.testing.assert_array_equal(aligned_series,
                                  np.array([[nan, 1., 2.],
                                            [0., 1., nan]]))
    assert zero_idx == 1


def test_align_series_unequal_length():
    t1 = np.array([1, 2])
    t2 = np.array([0, 1, 2])
    idx = [0, 1]
    series = [t1, t2]
    aligned_series, zero_idx = align_series(series, idx)
    np.testing.assert_array_equal(aligned_series,
                                  np.array([[nan, 1., 2.],
                                            [0., 1., 2.]]))
    assert zero_idx == 1

    t1 = np.array([1, 2, 3])
    t2 = np.array([0, 1])
    idx = [0, 1]
    series = [t1, t2]
    aligned_series, zero_idx = align_series(series, idx)
    np.testing.assert_array_equal(aligned_series,
                                  np.array([[nan, 1., 2., 3.],
                                            [0., 1., nan, nan]]))
    assert zero_idx == 1


def test_align_series_invalid_input():
    t1 = np.array([1, 2])
    t2 = np.array([0, 1])
    idx = [0, 1, 2]
    series = [t1, t2]
    with pytest.raises(ValueError):
        align_series(series, idx)

    t1 = np.array([1, 2])
    t2 = np.array([0, 1])
    idx = [0]
    series = [t1, t2]
    with pytest.raises(ValueError):
        align_series(series, idx)

    t1 = np.array([1, 2])
    t2 = np.array([0, 1])
    idx = [0, 2]
    series = [t1, t2]
    with pytest.raises(ValueError):
        align_series(series, idx)

    t1 = np.array([])
    t2 = np.array([0, 1])
    idx = [0, 1]
    series = [t1, t2]
    with pytest.raises(ValueError):
        align_series(series, idx)

    t1 = np.array([1, 2])
    t2 = np.array([0, 1])
    idx = [-1, 1]
    series = [t1, t2]
    with pytest.raises(ValueError):
        align_series(series, idx)


def test_align_series_idx_offset():
    t1 = np.array([1, 2])
    t2 = np.array([0, 1])
    idx = [1, 1]
    series = [t1, t2]
    aligned_series, zero_idx = align_series(series, idx)
    np.testing.assert_array_equal(aligned_series,
                                  np.array([[1., 2.],
                                            [0., 1.]]))
    assert zero_idx == 1


def test_align_series_single_element():
    t1 = np.array([1])
    idx = [0]
    series = [t1]
    aligned_series, zero_idx = align_series(series, idx)
    np.testing.assert_array_equal(aligned_series, np.array([[1.]]))
    assert zero_idx == 0


# ========================================
# ===== Test the embed function ==========
# ========================================

def test_embed_empty_list():
    arrays = []
    result = embed(arrays)
    assert result.shape == (0, 0)
    assert result.size == 0


def test_embed_single_array():
    arrays = [np.array([1, 2, 3])]
    result = embed(arrays)
    expected = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(result, expected)


def test_embed_mixed_lengths():
    arrays = [np.array([1, 2]), np.array([3, 4, 5])]
    result = embed(arrays)
    expected = np.array([[1.0, 2.0, np.nan],
                         [3.0, 4.0, 5.0]])
    np.testing.assert_array_equal(result, expected)


def test_embed_mixed_dtypes():
    arrays = [np.array([1, 2], dtype=int), np.array([3.5, 4.5, 5.5])]
    result = embed(arrays)
    expected = np.array([[1.0, 2.0, np.nan],
                         [3.5, 4.5, 5.5]])
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == float


def test_embed_with_zeros():
    arrays = [np.array([1, 2, 3]), np.array([0, 0])]
    result = embed(arrays)
    expected = np.array([[1.0, 2.0, 3.0],
                         [0.0, 0.0, np.nan]])
    np.testing.assert_array_equal(result, expected)


def test_embed_raises_on_non_1d_array():
    arrays = [np.array([[1, 2], [3, 4]])]
    with pytest.raises(ValueError, match="All arrays must be 1D"):
        embed(arrays)


# ========================================
# ===== Test the embed function ==========
# ========================================

def test_two_point_correlation_basic():
    ensemble = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    expected = np.array([
        [22., 26., 30.],
        [26., 31., 36.],
        [30., 36., 42.]
    ])

    result = two_point_correlation(ensemble, mean_removed=False, ignore_nan=False)
    assert result.shape == (3, 3)
    assert_array_almost_equal(result, expected)


def test_two_point_correlation_non_square_input():
    ensemble = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    expected = np.array([
        [8.5, 11.0, 13.5],
        [11.0, 14.5, 18.0],
        [13.5, 18.0, 22.5]
    ])

    result = two_point_correlation(ensemble, mean_removed=False, ignore_nan=False)
    assert result.shape == (3, 3)
    assert_array_almost_equal(result, expected)


def test_two_point_correlation_mean_removed():
    ensemble = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    expected = np.array([
        [6.0, 6.0, 6.0],
        [6.0, 6.0, 6.0],
        [6.0, 6.0, 6.0]
    ])

    result = two_point_correlation(ensemble, mean_removed=True, ignore_nan=False)
    assert result.shape == (3, 3)
    assert_array_almost_equal(result, expected)


def test_two_point_correlation_ignore_nan():
    ensemble = np.array([
        [1.0, 2.0, nan],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    expected = np.array([
        [22., 26., 43.5],
        [26., 31., 51.],
        [43.5, 51., 58.5]
    ])

    result = two_point_correlation(ensemble, mean_removed=False, ignore_nan=True)
    assert result.shape == (3, 3)
    assert_array_almost_equal(result, expected)


def test_two_point_correlation_many_nan_ignore():
    ensemble = np.array([
        [1.0, 2.0, nan],
        [4.0, 5.0, 6.0]
    ])
    expected = np.array([
        [8.5, 11.0, 24.0],
        [11.0, 14.5, 30.0],
        [24.0, 30.0, 36.0]
    ])

    result = two_point_correlation(ensemble, mean_removed=False, ignore_nan=True)
    assert result.shape == (3, 3)
    assert_array_almost_equal(result, expected)


def test_two_point_correlation_include_nan():
    ensemble = np.array([
        [1.0, 2.0, nan],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    expected = np.array([
        [22., 26., nan],
        [26., 31., nan],
        [nan, nan, nan]
    ])

    result = two_point_correlation(ensemble, mean_removed=False, ignore_nan=False)
    assert result.shape == (3, 3)
    assert_array_almost_equal(result, expected)


def test_two_point_correlation_nan_mean_removed():
    ensemble = np.array([
        [1.0, 2.0, np.nan],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    expected = np.array([
        [6., 6., 2.25],
        [6., 6., 2.25],
        [2.25, 2.25, 2.25]
    ])

    result = two_point_correlation(ensemble, mean_removed=True, ignore_nan=True)
    assert result.shape == (3, 3)
    assert_array_almost_equal(result, expected)


def test_two_point_correlation_empty_input():
    ensemble = np.array([])
    with pytest.raises(ValueError):
        two_point_correlation(ensemble, mean_removed=False, ignore_nan=False)


def test_two_point_correlation_non_2d_input():
    ensemble = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        two_point_correlation(ensemble, mean_removed=False, ignore_nan=False)


# ========================================
# ===== Test the is_toeplitz function ====
# ========================================

def test_is_toeplitz_square_toeplitz_matrix():
    # A classic Toeplitz matrix
    matrix = np.array([[1, 2, 3],
                       [4, 1, 2],
                       [5, 4, 1]])
    assert is_toeplitz(matrix)


def test_is_toeplitz_non_square_matrix():
    # A non-square matrix should raise a ValueError
    matrix = np.array([[1, 2, 3],
                       [4, 1, 2]])
    with pytest.raises(ValueError):
        is_toeplitz(matrix)


def test_is_toeplitz_square_non_toeplitz_matrix():
    # A square but non-Toeplitz matrix
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
    assert not is_toeplitz(matrix)


def test_is_toeplitz_single_element_matrix():
    # A 1x1 matrix is trivially Toeplitz
    matrix = np.array([[1]])
    assert is_toeplitz(matrix)


def test_is_toeplitz_invalid_input_not_array():
    # Input that is not a numpy array
    matrix = [[1, 2, 3], [4, 1, 2]]
    with pytest.raises(ValueError):
        is_toeplitz(matrix)


def test_is_toeplitz_non_2d_array():
    # Input is not 2D
    matrix = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        is_toeplitz(matrix)


def test_is_toeplitz_large_toeplitz_matrix():
    # A large Toeplitz matrix for performance check
    matrix = np.array([[1, 2, 3, 4],
                       [5, 1, 2, 3],
                       [6, 5, 1, 2],
                       [7, 6, 5, 1]])
    assert is_toeplitz(matrix)


def test_is_toeplitz_large_non_toeplitz_matrix():
    # A large non-Toeplitz matrix
    matrix = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])
    assert not is_toeplitz(matrix)
