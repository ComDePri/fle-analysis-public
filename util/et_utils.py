# Utility functions for eye tracking data analysis
import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import List, Tuple


def pursuit_onset(x, t, ltr=None):
    """
    Find the onset of pursuit in a trial.
    This assumes that the eye is at rest at the beginning of the trial, saccades to the target, and then starts
    pursuing the target.
    :param x: array like, horizontal eye position
    :param t: array like, time
    :param ltr: bool, left to right motion. If None, the function will try to infer the motion direction.
    :return: time in of the onset of pursuit, relative to trial start time
    """

    # Make sure t is increasing
    if np.any(np.diff(t) < 0):
        raise ValueError('t must be monotonically increasing.')
    # Make sure x and t have the same length
    if len(x) != len(t):
        raise ValueError('x and t must have the same length.')
    # Make sure x and t are of the same shape
    if x.shape != t.shape:
        raise ValueError('x and t must have the same shape.')
    # Make sure x and t are 1-dimensional
    if len(x.shape) > 1:
        raise ValueError('x and t must be 1-dimensional arrays.')

    t = np.array(t)
    x = np.array(x)
    t0 = min(t)
    diff = np.diff(x) / np.diff(t)
    diff = np.insert(diff, 0, np.nan)
    diff2 = np.diff(diff) / np.diff(t)
    diff2 = np.insert(diff2, 0, np.nan)
    if ltr is None:
        ltr = np.median(diff) > 0
    diff2 = diff2 if ltr else -diff2
    # If all elements of diff2 are NaN, return NaN
    if np.all(np.isnan(diff2)):
        return np.nan
    else:
        return t[np.nanargmax(diff2)] - t0


def trim_ramping_time(trial: pd.DataFrame, start_time_ms=200) -> pd.DataFrame:
    """
    Trim the initial ramping time from the trial data.

    Parameters:
    trial (pd.DataFrame): The trial data containing a 'trial_time' column.
    start_time_ms (int): The start time in milliseconds to trim from (default 200).

    Returns:
    pd.DataFrame: The trimmed trial data.
    """
    return trial[trial['trial_time'] >= start_time_ms]


def remove_short_segments(series: pd.Series, min_length: int) -> pd.Series:
    """
    Set segments of True values shorter than min_length to False.

    Parameters:
    series (pd.Series): The boolean series to process.
    min_length (int): The minimum length of True segments to retain.

    Returns:
    pd.Series: The processed series with short segments set to False.
    """
    groups = (series != series.shift()).cumsum()
    lengths = series.groupby(groups).transform('size')
    return series.where(~((series == True) & (lengths < min_length)), other=False)


def split(df: pd.DataFrame, col: str) -> List[pd.DataFrame]:
    """
    Split a DataFrame into a list of DataFrames, each containing a single segment of True values.

    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    col (str): The column with boolean values to use for splitting.

    Returns:
    List[pd.DataFrame]: A list of DataFrames, each containing a single segment of True values.
    """
    groups = (df[col] != df[col].shift()).cumsum()
    return [group for _, group in df.groupby(groups) if group[col].iloc[0]]


def trim_after_event(df: pd.DataFrame, event_col='flasher') -> pd.DataFrame:
    """
    Remove all rows after the first True value in the specified event column.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    event_col (str): The boolean column indicating the event.

    Returns:
    pd.DataFrame: The DataFrame trimmed after the first True event.
    """
    event_indices = df[df[event_col] == 1].index
    if event_indices.empty:
        return df
    return df.loc[:event_indices[0]]


def fit_seg(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """
    Fit a line to the data and return the slope and intercept.

    Parameters:
    x (pd.Series): The x-values of the data.
    y (pd.Series): The y-values of the data.

    Returns:
    Tuple[float, float]: The slope and intercept of the fitted line.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept


def inflate_true_segments(series: pd.Series, inflation: int) -> pd.Series:
    """
    Inflate continuous segments of True values in a boolean pandas Series by a given number of items on each side.

    :param series: A pandas Series with boolean values (True/False).
    :param inflation: An integer indicating how many items to inflate on each side of True segments.
    :returns: A pandas Series with the inflated True segments.
    :raises ValueError: If the input series is not of boolean type.
    """
    if series.empty:
        return series

    if not pd.api.types.is_bool_dtype(series):
        raise ValueError(f"The input series must be of boolean type, but got {series.dtype}.")

    inflated = series.copy()
    true_indices = series.index[series]
    for idx in true_indices:
        start_idx = max(series.index[0], idx - inflation)
        end_idx = min(series.index[-1], idx + inflation)
        inflated.loc[start_idx:end_idx] = True
    return inflated


def identify_tracking_epochs(trial: pd.DataFrame,
                             slope_thresh: float,
                             ramp_time: int = 200,
                             window: int = 11,
                             v_thresh: float = 2000,
                             inflation: int = 15,
                             min_length: int = 50) -> List[pd.DataFrame]:
    """
    Identify epochs of tracking in a DataFrame.

    :param trial: The trial data containing eye tracking information.
    :param slope_thresh: The slope threshold for identifying tracking epochs.
    :param ramp_time: The initial ramping time to trim (default 200 ms).
    :param window: The window size for smoothing velocity (default 11).
    :param v_thresh: The velocity threshold for identifying saccades (default 2000).
    :param inflation: The inflation factor for suspected tracking segments (default 15).
    :param min_length: The minimum length of tracking epochs to retain (default 50 timepoints).

    :returns List[pd.DataFrame]: A list of DataFrames, each containing a tracking epoch.
    """

    tri = trial.copy()
    tri['trial_time'] = tri['timestamp_ms'] - tri['timestamp_ms'].min()

    # Trim the trial data
    trimmed_trial = trim_ramping_time(tri, ramp_time)
    trimmed_trial = trim_after_event(trimmed_trial, 'flasher')

    # Identify suspected saccades and blinks
    trimmed_trial['v_x_smooth'] = epoch_v(trimmed_trial, window)
    trimmed_trial['saccade'] = trimmed_trial['v_x_smooth'].abs() > v_thresh
    trimmed_trial['saccade_inf'] = inflate_true_segments(trimmed_trial['saccade'], inflation)
    trimmed_trial['blink'] = trimmed_trial['x_pix'].isna() | trimmed_trial['y_pix'].isna()

    # Remove short segments of suspected tracking
    blink_or_saccade = trimmed_trial['saccade_inf'] | trimmed_trial['blink']
    not_blink_or_saccade = ~blink_or_saccade
    short_segments_removed = remove_short_segments(not_blink_or_saccade, min_length)
    trimmed_trial['suspected_tracking'] = short_segments_removed

    # Split the trial into epochs, fit a line to each epoch, and exclude epochs with slope below threshold
    epochs = split(trimmed_trial, 'suspected_tracking')
    fits = [fit_seg(epoch['trial_time'] / 1000, epoch['x_pix_ltr_reflect']) for epoch in epochs]
    tracking_epochs = [epoch for epoch, (slope, _) in zip(epochs, fits) if slope > slope_thresh]

    return tracking_epochs


def epoch_v(epoch, window=21, assume_equidistant=False):
    """Return the speed of the epoch in pixels per second.

    :param epoch: The epoch data containing 'x_pix_ltr_reflect' and 'timestamp_ms' columns.
    :param window: The window size for smoothing velocity (default 21).
    :param assume_equidistant: Whether to assume equidistant timestamps (default False). If True, increases efficiency.
    :returns: The smoothed velocity of the epoch in pixels per second.
    """
    if assume_equidistant:
        dt = epoch['timestamp_ms'][1] - epoch['timestamp_ms'][0]
        v_x = epoch['x_pix_ltr_reflect'].diff() / dt * 1000
    else:
        v_x = epoch['x_pix_ltr_reflect'].diff() / epoch['timestamp_ms'].diff() * 1000
    if window == 1:
        return v_x
    else:
        return v_x.rolling(window=window, center=True).mean()
