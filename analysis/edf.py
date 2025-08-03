# Utility functions to read eyetracking EDF files

import re
import pandas as pd
import os
import subprocess
import numpy as np
from tqdm import tqdm


def edf2asc(file, force=False):
    """
    Convert an EDF file to an ASC file. Requires edf2asc to be installed and in the PATH. Skips conversion if an ASC
    file already exists.
    :param force: If True, force conversion even if an ASC file already exists (overwrites the ASC file).
    :param file: The path to the EDF file.
    :return: The path to the ASC file.
    """
    if not file.endswith('.edf'):
        raise ValueError('edf2asc() only works on EDF files.')
    root, _ = os.path.splitext(file)
    asc_file = root + '.asc'
    if not os.path.isfile(asc_file) or force:
        print('Converting ' + file + ' to ASC...')
        subprocess.run(['edf2asc', file])
    else:
        print('Skipping ' + file + ' - already has .ASC...')
    return asc_file


def read_asc(path, save=False) -> (pd.DataFrame, pd.DataFrame):
    """
    Read an ASC file and return a list of pandas DataFrames, one for each trial.
    :param save: Whether to save the trials to disk.
    :param path: The path to the ASC file.
    :return: Two pandas DataFrames, one for trials and one for faux trials. The dataframes have the following columns:
        this_n: The trial number, exactly as it appears in the ASC file (and corresponding to results.csv).
        premature_multiplicity: The repetition of this trial due to premature responses (0 for non-premature trials).
        timestamp_ms: The timestamp of the sample in milliseconds.
        x_pix: The x position of the gaze in pixels.
        y_pix: The y position of the gaze in pixels.
        pupil_size: The pupil size in arbitrary units.
        flasher: Whether the flasher was on or off.
        flags: A string of flags.
    """

    # TODO: parse without regex to improve performance.
    trial_delimiter = re.compile(r'^MSG\s+'
                                 r'(?P<time>\d+)\s+'
                                 r'TRIALID\s+'
                                 r'(?P<trial_id>\d+)\s+'
                                 r'(?P<state>BEGIN|END)$')
    faux_trial_delimiter = re.compile(r'^MSG\s+'
                                      r'(?P<time>\d+)\s+'
                                      r'TRIALID blank\s+'
                                      r'(?P<state>BEGIN|END)$')
    flasher_delimiter = re.compile(r'^MSG\s+'
                                   r'(?P<time>\d+)\s+'
                                   r'flasher_(?P<state>on|off)$')
    sample_pattern = re.compile(r'^(?P<time>\d+)\s+'
                                r'(?P<x>\d*\.\d*|\d+\.?\d*)\s+'
                                r'(?P<y>\d*\.\d*|\d+\.?\d*)\s+'
                                r'(?P<pupil_size>\d*\.\d+|\d+\.?\d*)\s+'
                                r'(?P<flags>\S*)$')
    trials = pd.DataFrame(
        columns=['this_n', 'premature_multiplicity', 'timestamp_ms', 'x_pix', 'y_pix', 'pupil_size',
                 'flasher', 'flags'])
    trials = trials.astype({'this_n': 'int16', 'premature_multiplicity': 'int8', 'timestamp_ms': 'int32',
                            'x_pix': 'float16', 'y_pix': 'float16', 'pupil_size': 'float16',
                            'flasher': 'bool', 'flags': 'str'})
    faux_trials = trials.copy()
    trial_lines = []
    all_trials = []
    all_faux_trials = []

    flasher_state = False
    this_n = None
    multiplicity = 0
    this_n_faux = 0
    in_trial = False
    with open(path) as f:
        nlines = sum(1 for _ in f)
    with open(path) as f:
        for line in tqdm(f, total=nlines, desc='Reading ASC file'):
            trial_match = trial_delimiter.match(line)
            faux_trial_match = faux_trial_delimiter.match(line)
            flasher_match = flasher_delimiter.match(line)
            sample_match = sample_pattern.match(line)

            if trial_match:
                if trial_match.group('state') == 'BEGIN':
                    in_trial = True
                    new_n = int(trial_match.group('trial_id'))
                    multiplicity = multiplicity + 1 if new_n == this_n else 0
                    this_n = new_n
                    n_to_write = this_n
                elif trial_match.group('state') == 'END':
                    all_trials.extend(trial_lines)
                    trial_lines = []
                    in_trial = False
            elif faux_trial_match:
                if faux_trial_match.group('state') == 'BEGIN':
                    this_n_faux += 1
                    multiplicity = 0
                    n_to_write = this_n_faux
                    in_trial = True
                elif faux_trial_match.group('state') == 'END':
                    all_faux_trials.extend(trial_lines)
                    trial_lines = []
                    in_trial = False
            elif flasher_match:
                flasher_state = flasher_match.group('state') == 'on'
            elif sample_match and in_trial:
                timestamp = np.int32(sample_match.group('time'))
                # Note: presence of np.nan upcasts the dtype to float64.
                x = np.float16(sample_match.group('x')) if not sample_match.group('x') == "." else np.nan
                y = np.float16(sample_match.group('y')) if not sample_match.group('y') == "." else np.nan
                pupil_size = np.float16(sample_match.group('pupil_size'))
                flags = sample_match.group('flags')

                sample_data = {'this_n': np.int16(n_to_write), 'premature_multiplicity': np.int16(multiplicity),
                               'timestamp_ms': timestamp,
                               'x_pix': x, 'y_pix': y, 'pupil_size': pupil_size,
                               'flasher': flasher_state, 'flags': flags}
                trial_lines.append(sample_data)
    trials = pd.DataFrame(all_trials).astype(trials.dtypes)
    faux_trials = pd.DataFrame(all_faux_trials).astype(faux_trials.dtypes)
    if save:
        root, _ = os.path.splitext(path)
        trials.reset_index().to_feather(root + '_trials.feather')
        faux_trials.reset_index().to_feather(root + '_faux.feather')
    return trials, faux_trials
