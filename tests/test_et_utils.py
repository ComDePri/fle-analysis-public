import pytest
import pandas as pd
from util.et_utils import inflate_true_segments


def test_basic_inflation():
    # Test with a simple case where inflation is 1
    series = pd.Series([False, False, True, True, False, False])
    result = inflate_true_segments(series, 1)
    expected = pd.Series([False, True, True, True, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_no_inflation():
    # Test with an inflation of 0 (should not change the result)
    series = pd.Series([False, False, True, False])
    result = inflate_true_segments(series, 0)
    expected = pd.Series([False, False, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_inflation_larger_than_series():
    # Test where inflation extends beyond series boundaries
    series = pd.Series([True, False, False])
    result = inflate_true_segments(series, 2)
    expected = pd.Series([True, True, True])
    pd.testing.assert_series_equal(result, expected)


def test_single_true():
    # Test with a single True value in the series
    series = pd.Series([False, False, True, False, False])
    result = inflate_true_segments(series, 1)
    expected = pd.Series([False, True, True, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_all_false():
    # Test with a series of all False values
    series = pd.Series([False, False, False])
    result = inflate_true_segments(series, 1)
    expected = pd.Series([False, False, False])
    pd.testing.assert_series_equal(result, expected)


def test_all_true():
    # Test with a series of all True values
    series = pd.Series([True, True, True])
    result = inflate_true_segments(series, 1)
    expected = pd.Series([True, True, True])
    pd.testing.assert_series_equal(result, expected)


def test_empty_series():
    # Test with an empty series
    series = pd.Series([], dtype=bool)
    result = inflate_true_segments(series, 1)
    expected = pd.Series([], dtype=bool)
    pd.testing.assert_series_equal(result, expected)


def test_inflation_with_gaps():
    # Test where True values are not continuous
    series = pd.Series([True, False, True, False])
    result = inflate_true_segments(series, 1)
    expected = pd.Series([True, True, True, True])
    pd.testing.assert_series_equal(result, expected)


def test_non_boolean_series():
    # Test with a non-boolean series, should raise a ValueError
    series = pd.Series([1, 0, 1, 0])
    with pytest.raises(ValueError):
        inflate_true_segments(series, 1)
