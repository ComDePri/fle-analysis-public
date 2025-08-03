"""
Calculate the maximum likelihood estimate (MLE) from log-likelihood files.

This script loads log-likelihood files from a specified directory, calculates the MLE for each participant and speed condition, and saves the results to a CSV file.
Usage:
    python calc_mle_from_llk.py <llk_path> [--output_path <output_path>]

Arguments:
    llk_path: Path to the directory containing the log-likelihood files.
    --output_path: Path to save the MLE DataFrame. Defaults to the current directory.

Example:
    python calc_mle_from_llk.py /path/to/llk_files --output_path /path/to/save/mle
"""


import os
import argparse
import pandas as pd
import xarray as xr
from util.xarray import coordmax
from analysis.gaussian_process import marginalize_lse
from tqdm import tqdm


def calc_mle_df(llk_dict):
    """
    Calculate the MLE for each participant and speed condition in the data_dict.

    :param llk_dict: dict keyed by part_id, containing a dict keyed by speed, containing an xarray representing the log-likelihood of the parameters.
    :return: DataFrame with columns part_id, speed, and the MLE values for each parameter.
    """

    mle = {}
    for part_id, part_llks in tqdm(llk_dict.items(), leave=True, position=0,
                                   desc="Calculating MLE"):
        for v_t, v_llk in part_llks.items():
            total_llk = v_llk.sum('epoch')
            marginal_llk = marginalize_lse(total_llk, ['sig_y'])
            mle.setdefault(part_id, {})[v_t] = coordmax(marginal_llk)

    df = pd.DataFrame([
        {'part_id': part_id, 'speed': v_t, **mle_val}
        for part_id, part_mle in mle.items()
        for v_t, mle_val in part_mle.items()
    ])
    df.set_index(['part_id', 'speed'], inplace=False)
    return df


def load_llks(path):
    """
    Load log-likelihood files from the specified directory.

    :param path: Path to the directory containing the log-likelihood files.
    :return: Dictionary keyed by part_id, containing a dict keyed by speed, containing an xarray representing the log-likelihood of the parameters.
    """

    data = {}
    for root, _, files in os.walk(path):
        if 'skipme' in root:
            continue
        for file in files:
            if file.endswith('.nc'):
                llk = xr.load_dataarray(os.path.join(root, file))
                data.setdefault(llk.attrs['part_id'], {})[llk.attrs['v_t']] = llk
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('llk_path', type=str, help='Path to the directory containing the log-likelihood files.')
    parser.add_argument('--output_path', type=str, help='Path to save the MLE DataFrame.', default="")
    args = parser.parse_args()

    print("Loading log-likelihoods...")
    llks = load_llks(args.llk_path)
    print("Calculating MLE...")
    mle_df = calc_mle_df(llks)
    savepath = os.path.join(args.output_path, "mle.csv")
    print(f"Saving MLE to {savepath}")
    mle_df.to_csv(savepath)
    print("Done.")
