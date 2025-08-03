import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from tqdm import tqdm
import os


def threshold_to_halfpoint(threshold, slope, scale):
    """
    Converts a threshold parameter to the stimulus level at which the psychometric function is equal 1/2.
    Assumes a Weibull function.

    :param threshold: threshold of the psychometric function
    :param slope: slope of the psychometric function
    :param scale: scale of the input to the psychometric function
    :return:
    """
    if scale == 'linear':
        return threshold * np.log(2) ** (1 / slope)
    elif scale == 'log10':
        return threshold + (1 / slope) * np.log10(np.log(2))
    elif scale == 'dB':
        return threshold + (20 / slope) * np.log10(np.log(2))
    else:
        raise ValueError('Invalid scale, only \'linear\', \'log10\' and \'dB\' are supported.')


def halfpoint_to_threshold(halfpoint, slope, scale):
    """
    Converts a halfpoint parameter to the threshold parameter of the psychometric function.
    Assumes a Weibull function.

    :param halfpoint: stimulus level at which the psychometric function is equal 1/2
    :param slope: slope of the psychometric function
    :param scale: scale of the input to the psychometric function
    :return:
    """
    if scale == 'linear':
        return halfpoint / (np.log(2) ** (1 / slope))
    elif scale == 'log10':
        return halfpoint - (1 / slope) * np.log10(np.log(2))
    elif scale == 'dB':
        return halfpoint - (20 / slope) * np.log10(np.log(2))
    else:
        raise ValueError('Invalid scale, only \'linear\', \'log10\' and \'dB\' are supported.')


def calc_sd(pdf, quant):
    """
    Calculates the standard deviation of a function of a distribution.

    :param pdf: probability density function defined on an n-dimensional array
    :param quant: quantity defined over the same domain as pdf, to calculate the standard deviation of,
    given as an n-dimensional array
    :return: standard deviation of func with respect to the pdf
    """
    # TODO figure out if I should use a numerical integration instead of summation.
    if np.equal(pdf.shape, quant.shape).all():  # TODO can relax this condition to allow for broadcasting, test first
        first_moment = (pdf * quant).sum()
        second_moment = (pdf * quant ** 2).sum()
        var = second_moment - first_moment ** 2
        sd = np.sqrt(var)
        return sd
    else:
        raise ValueError('The pdf and the quantity must have the same shape.')


def hdi(pdf, domain, alpha=0.05) -> list:
    """
    Calculates the highest density interval of a distribution.
    :param pdf: Probability density function defined on a 1-dimensional array.
    :param domain: Domain of the pdf, defined on a 1-dimensional array. pdf[i] is the probability of domain[i].
    :param alpha: 1-alpha is the HDI. Default is 0.05.
    :return: The HDI of the distribution. A list of tuples, each tuple containing the lower and upper bound of an HDI.
    """
    # Asserts
    assert len(pdf.shape) == 1, 'pdf must be a 1-dimensional array.'
    assert len(domain.shape) == 1, 'domain must be a 1-dimensional array.'
    assert pdf.shape == domain.shape, 'pdf and domain must have the same shape.'
    assert 0 < alpha < 1, 'alpha must be between 0 and 1.'
    # First, we need to find the threshold probability which defines the HDI.
    threshold = hdi_thresh(pdf, alpha)
    hdi_region = np.where(pdf >= threshold, True, False)
    lowers = []
    uppers = []
    for i in range(len(hdi_region)):
        if hdi_region[i]:
            if i == 0 or not hdi_region[i - 1]:
                lowers.append(domain[i])
            if i == len(hdi_region) - 1 or not hdi_region[i + 1]:
                uppers.append(domain[i])
    edges = [(l, u) for l, u in zip(lowers, uppers)]
    return edges


def hdi_thresh(dist, alpha=0.05):
    """
    Calculates the threshold of the highest density interval of a distribution.
    This is a helper function for hdi, but can also be used independently (for contour levels, for example).
    :param dist: Distribution to calculate the HDI threshold of.
    :param alpha: 1-alpha is the HDI. Default is 0.05.
    :return: The threshold of the HDI of the distribution.
    """
    if len(dist.shape) > 1:
        dist_flat = dist.flatten()
    else:
        dist_flat = dist
    dist_sorted = np.sort(dist_flat)
    dist_sorted = dist_sorted[::-1]  # reverse the array
    cumsum = np.cumsum(dist_sorted)
    threshold_idx = np.where(cumsum >= 1 - alpha)[0][0]
    threshold = dist_sorted[threshold_idx]
    return threshold


def regions_to_yerr(point_estimates, regions):
    """
    Converts a list of uncertainty regions (HDI, CI, etc.) to a list of yerr values for plotting.
    :param point_estimates: List of point estimates.
    :param regions: List of HDI regions.
    :return: 2-by-N ndarray of yerr values. Can be passed to the yerr argument of plt.errorbar alongside point_estimates.
    """
    assert len(point_estimates) == len(regions), 'point_estimates and regions must have the same length.'
    yerr = []
    for est, region in zip(point_estimates, regions):
        if len(region) == 1:
            bounds = region[0]
            yerr.append((est - bounds[0], bounds[1] - est))
        else:
            raise ValueError(f'regions[{point_estimates.index(est)}] has more than one uncertainty region.')
    yerr = np.asarray(yerr)
    for err in yerr.flatten():
        assert err >= 0, 'Some yerr values are negative. Make sure point_estimates are inside their regions.'
    yerr = np.transpose(yerr)
    return yerr


def apply_to_dir(root, ext, func, *args, **kwargs):
    """
    Apply a function to all files in a directory with a given extension, recursively
    :param root: Directory to search.
    :param ext: Extension of files to apply the function to. If None, applies to all files.
    :param func: Function to apply to each file.
    :param args: Positional arguments to pass to func.
    :param kwargs: Keyword arguments to pass to func.
    :return: List of outputs of func.
    """
    output = []
    for root, dirs, files in os.walk(root):
        for filename in tqdm(files):
            if ext is None or filename.endswith(ext):
                filepath = os.path.join(root, filename)
                output.append(func(filepath, *args, **kwargs))
    return output


def two_point_correlation(
        ensemble: NDArray[np.float64],
        mean_removed: bool = True,
        ignore_nan: bool = False
) -> NDArray[np.float64]:
    """
    Calculate the two-point correlation function of an ensemble of time-series.

    :param ensemble: a 2D array, where each row is a time-series. Shape: (n_series, n_points)
    :param mean_removed: If True, subtract the global mean from each time-series before calculating the correlation.
    :param ignore_nan: If True, ignore NaN values when calculating the correlation.
    :return: A 2d array of the correlation matrix. Shape: (n_points, n_points)

    """
    if ensemble.ndim != 2 or ensemble.size == 0:
        raise ValueError("Input 'ensemble' must be a non-empty 2D array.")

    if ignore_nan:
        mean = np.nanmean
    else:
        mean = np.mean

    if mean_removed:
        ensemble = ensemble - mean(ensemble, axis=0)
    res = mean(ensemble[:, :, np.newaxis] * ensemble[:, np.newaxis, :], axis=0, dtype=np.float64)
    return res


def reflect(x, x_0):
    """
    Reflects a value around a point.
    :param x: Value to reflect.
    :param x_0: Point to reflect around.
    :return: Reflected value.
    """
    return 2 * x_0 - x


def align_series(series: List[np.ndarray], zero_idx: List[int]) -> Tuple[np.ndarray, int]:
    """
    Align a list of time-series to a common zero point. Pads with NaNs to align the series.

    :param series: a list of 1D arrays, not necessarily the same length
    :param zero_idx: a matching list of indices to align the series to
    :return: a tuple containing:
        - A 2D NumPy array of the aligned series
        - An integer representing the index of the common zero point (based on zero_idx)
    :raises: ValueError if series and zero_idx have different lengths
    :raises: ValueError if an index in zero_idx is out of bounds for the corresponding series
    """
    if len(series) != len(zero_idx):
        raise ValueError(f'series and zero_idx must have the same length, got {len(series)} and {len(zero_idx)}.')
    for i, (s, ind) in enumerate(zip(series, zero_idx)):
        if not (0 <= ind < len(s)):
            raise ValueError(f'Index {ind} in zero_idx at position {i} is out of bounds for the corresponding series.')

    max_len = max([len(s) for s in series])
    zero_idx = np.asarray(zero_idx)
    rel_idx = zero_idx - zero_idx.min()
    max_rel = rel_idx.max()

    # Pad each series with the pad value
    aligned_series = []
    for s, ind in zip(series, rel_idx):
        left_pad = max_rel - ind  # Need to pad on the left with max_rel. Subtract ind to "roll" the series.
        right_pad = max_len - len(s) + ind  # Need to pad on the right with max_len - len(s), we add ind to "roll" the series.
        pad_width = (left_pad, right_pad)
        if s.dtype != float:
            s = s.astype(float)
        aligned_series.append(np.pad(s, pad_width, 'constant', constant_values=np.nan))
    aligned_array = np.vstack(aligned_series)

    # remove columns that are completely NaN padded
    valid_columns = ~np.all(np.isnan(aligned_array), axis=0)
    aligned_array = aligned_array[:, valid_columns]
    return aligned_array, zero_idx.max()


def embed(arrays: List[np.ndarray]) -> np.ndarray:
    """
    Embeds a jagged list of 1d arrays into a single 2D array, padding with NaNs.
    :param arrays: A list of 1D arrays.
    :return: A 2d NumPy array with the arrays embedded.
    :raises ValueError: If any of the arrays is not 1D.
    """

    if not arrays:
        return np.empty((0, 0))

    for a in arrays:
        if a.ndim != 1:
            raise ValueError('All arrays must be 1D.')

    for a in arrays:
        if a.dtype != float:
            a = a.astype(float)

    max_len = max([len(a) for a in arrays])
    padded_arrays = [
        np.pad(a.astype(float) if not np.issubdtype(a.dtype, np.floating) else a,
               (0, max_len - len(a)),
               'constant', constant_values=np.nan)
        for a in arrays]
    return np.vstack(padded_arrays)


def is_toeplitz(matrix):
    """
    Check if a matrix is a Toeplitz matrix (constant values along all diagonals).

    Parameters:
    -----------
    matrix : np.ndarray
        The matrix to check. Should be 2D and square for a valid Toeplitz check.

    Returns:
    --------
    bool
        True if the matrix is Toeplitz, False otherwise.

    Raises:
    -------
    ValueError
        If the input is not a 2D square matrix.

    Notes:
    ------
    The check takes O(n^2) time for an n x n matrix, making it efficient for large matrices.
    """
    # Validate input matrix
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2D.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Check if all diagonals are constant
    for k in range(0, matrix.shape[0]):
        if not np.all(np.diag(matrix, k=k) == np.diag(matrix, k=k)[0]) or \
                not np.all(np.diag(matrix, k=-k) == np.diag(matrix, k=-k)[0]):
            return False
    return True


