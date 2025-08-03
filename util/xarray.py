import xarray as xr
import numpy as np


def coordmax(da: xr.DataArray) -> dict:
    """Find the coordinates of the maximum value in a DataArray.

    :param da: DataArray to find the maximum value in.
    :return: Dictionary of the coordinates of the maximum value, indexed by dimension name.
    """
    idx = da.argmax(dim=da.dims)
    return {dim: da.coords[dim].values[idx[dim]] for dim in da.dims}


def bootstrap(da: xr.DataArray,
              sample_dim: str,
              statistic,
              n_resamples: int = 1000,
              **statistic_kwargs) -> xr.DataArray:
    """Generate a bootstrapped sample of a DataArray along the specified dimension.

    :param da: DataArray to generate the bootstrapped sample from.
    :param sample_dim: Dimension to resample along.
    :param statistic: Function to apply to each bootstrapped sample.
    :param n_resamples: Number of bootstrap samples to generate.
    :param statistic_kwargs: Additional keyword arguments to pass to the statistic function.
    :return: Bootstrapped sample of the DataArray.
    """

    is_method = hasattr(da, statistic) and callable(getattr(da, statistic))
    size = da.sizes[sample_dim]
    results = []

    for _ in range(n_resamples):
        idx = np.random.randint(0, size, size=size)
        sample = da.isel({sample_dim: idx})
        if is_method:
            res = getattr(sample, statistic)(**statistic_kwargs)
        else:
            res = statistic(sample, **statistic_kwargs)
        results.append(res)

    return xr.concat(results, dim='resample', join='override')
