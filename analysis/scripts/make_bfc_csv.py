"""
This script creates a single csv dataframe from the binary forced-choice data of a single experiment
and the attention check data. It loads the data from the specified directory, processes it, and saves it to two csv files:

- bfc_df.csv: Contains the binary forced-choice data.
- attn_check_df.csv: Contains the attention check data.

The script uses the DataAnalyzer class to load the data and process it into a DataFrame.

Usage:
    python make_bfc_csv.py <data_path> [--output_path <output_path>]

Arguments:
    data_path: Path to the directory containing the data files.
    --output_path: Path to save the DataFrame. Defaults to the current directory.

Example:
    python make_bfc_csv.py /path/to/data --output_path /path/to/save
"""

import os
import argparse
import pandas as pd
from analysis.DataAnalyzer import DataAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='Path to the directory containing the data files.')
    parser.add_argument('--output_path', type=str, help='Path to save the DataFrame.', default="")
    args = parser.parse_args()

    print("Loading data...")
    analyzer = DataAnalyzer(args.data_path, args.data_path)
    analyzer.load_data()
    print("Creating DataFrame...")
    df = pd.concat({str(part_id): analyzer.data[part_id]['results'] for part_id in analyzer.participants},
                   names=['part_id'])
    df = df.drop(['thisRow.t',
                  'notes',
                  'exp_time_sec',
                  'mean_est',
                  'mean_hdi',
                  'speed',
                  'thisRepN',
                  'nTrials'], axis="columns")
    df = df.reset_index()
    df = df.rename(columns={'thisN': 'this_n',
                            'thisRepN': 'this_rep_n',
                            'intensity': 'offset'})
    savepath = os.path.join(args.output_path, "bfc_df.csv")
    print(f"Saving DataFrame to {savepath}")
    df.to_csv(savepath, index=False)

    # do the same for attention check data. Explicitly add a column for part_id
    print("Creating attention check DataFrame...")
    df = pd.concat({str(part_id): analyzer.data[part_id]['attn_check'] for part_id in analyzer.participants},
                   names=['part_id'])
    # cast part_id to string
    df.index = df.index.set_levels([df.index.levels[0].astype(str), df.index.levels[1]])
    df = df.reset_index()
    df = df.drop(['level_1'], axis="columns")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis="columns")
    savepath = os.path.join(args.output_path, "attn_check_df.csv")
    print(f"Saving DataFrame to {savepath}")
    df.to_csv(savepath, index=False)
    print("Done.")
