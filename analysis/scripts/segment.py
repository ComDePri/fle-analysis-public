"""
segment.py
This script segments eye tracking data into tracking epochs based on the speed of the eye movements.
Usage:
    python segment.py <data_dir> [--test]
Arguments:
    <data_dir>       Path to the data directory containing the eye tracking data.
    --test           Run in test mode, loading only two participants
Options:
    -h, --help       Show this help message and exit
Description:
    This script loads eye tracking data from the specified directory, segments the data into epochs based on the speed of the eye movements,
    and saves the segmented epochs to disk. The segmentation parameters are also saved to a JSON file.
    The segmentation is done by identifying tracking epochs based on the speed of the eye movements.
    The parameters for segmentation can be adjusted in the PARAMS dictionary.
"""


import argparse
import os
import logging
import sys
import json

from analysis.DataAnalyzer import DataAnalyzer
from util.et_utils import identify_tracking_epochs

PARAMS = {
    'slope_thresh_factor': 0.5,  # the slope of the linear fit must be greater than this times the trial speed
    'ramp_time': 200,  # ignore the beginning of the trial
    'window': 11,  # smoothing window for the velocity
    'saccade_speed_thresh': 2000,  # speed over this is considered a saccade
    'inflation': 15,  # include this many time points before and after the saccade
    'min_length': 50,  # reject epochs shorter than this
}


def slope_thresh(speed):
    return speed * PARAMS['slope_thresh_factor']


def segment_for_speed(speed, et_trials, bfc_results, seg_params):
    """
    Segment the eye tracking data for a given speed.

    :param speed: The speed of the eye movement.
    :param et_trials: The eye tracking trials data.
    :param bfc_results: The results of the binary forced choice task.
    :param seg_params: The segmentation parameters.
    :return: A list of segmented epochs.
    """

    params = seg_params.copy()
    params['slope_thresh'] = slope_thresh(speed)
    params['v_thresh'] = params['saccade_speed_thresh']
    params.pop('slope_thresh_factor')
    params.pop('saccade_speed_thresh')
    epochs = []
    for n, trial in et_trials.groupby('this_n'):
        if len(trial.dropna()) == 0:
            continue
        if (n, speed) in bfc_results.index:
            epochs.extend(identify_tracking_epochs(trial, **params))
    return epochs


def segment_for_part(da, part_id, seg_params):
    """
    Segment the eye tracking data for a given participant.

    :param da: The DataAnalyzer object containing the data.
    :param part_id: The participant ID.
    :param seg_params: The segmentation parameters.
    :return: A dictionary of segmented epochs for each speed, keyed by speed.
    """

    trials = da.data[part_id]['et_trials']
    twoAFCres = da.data[part_id]['results']
    speeds = da.data[part_id]['params']['speeds']
    epochs = {}
    for speed in speeds:
        epochs[speed] = segment_for_speed(speed, trials, twoAFCres, seg_params)
    return epochs


def run():
    """
    Main function to run the segmentation script.

    This function sets up the command line argument parsing, loads the data,
    segments the data into epochs, and saves the segmented epochs to disk.
    It also saves the segmentation parameters to a JSON file.
    The segmentation is done by identifying tracking epochs based on the speed of the eye movements.
    The parameters for segmentation can be adjusted in the PARAMS dictionary.

    :return: None
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Segment eye tracking data.')
    parser.add_argument('data_dir', type=str, help='Path to the data directory.')
    parser.add_argument('--test', action='store_true', help='Test mode.')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    proj_dir = data_dir
    da = DataAnalyzer(data_dir=data_dir, proj_dir=proj_dir)
    logger.info(f"Loading data from {data_dir}")
    da.load_data(test=args.test)

    logger.info("Reflecting trials")
    da.reflect_trials()

    logger.info("Segmenting trials...")
    part_epochs = {}
    for part_id in da.participants:
        logger.info(f"Segmenting data for participant {part_id}")
        # skip if already exists
        if os.path.exists(f"{da.data_dir}/processed/epochs/{part_id}"):
            logger.info(f"Participant {part_id} already segmented. Skipping.")
            continue
        tracking_epochs = segment_for_part(da, part_id, PARAMS)
        part_epochs[part_id] = tracking_epochs

    logger.info("Saving segmented epochs to disk")
    for part, data in part_epochs.items():
        logger.info(f"Saving participant {part}")
        part_dir = f"{da.data_dir}/processed/epochs/{part}"
        os.makedirs(part_dir, exist_ok=True)
        for speed, epochs in data.items():
            speed_dir = f"{part_dir}/{speed}"
            os.makedirs(speed_dir, exist_ok=True)
            for i, epoch in enumerate(epochs):
                epoch.to_csv(f"{speed_dir}/{i:>03}.csv", index=False)
    logger.info("Saving segmentation parameters to file")
    with open(f"{da.data_dir}/processed/epoch_segmentation_params.json", 'w') as f:
        json.dump(PARAMS, f, indent=4)

    logger.info("Done.")


if __name__ == '__main__':
    run()
