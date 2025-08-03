import json
import os
import warnings
from typing import List, Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.optimize import curve_fit
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from questplus.psychometric_function import weibull, norm_cdf
from util.qp_extend import QuestPlusNorm
from tqdm import tqdm
from tqdm.contrib import tzip
from util.FLE_utils import hdi, regions_to_yerr, hdi_thresh, apply_to_dir, reflect
import ast
from typing import Union
from util.motion import constant1d
from util.et_utils import pursuit_onset
from ComDePy.viz import set_defaults, tile_subplots

set_defaults()
mpl.rcParams['axes.grid'] = True


def my_norm_CDF(x, mean, sd):
    """
    A wrapper for the QuestPlusNorm norm_cdf function.
    :param x: the x value to evaluate the function at
    :param mean: the mean of the Normal distribution
    :param sd: the standard deviation of the Normal distribution
    :return: $\phi(x; \mu, \sigma)$
    """
    return norm_cdf(intensity=x, mean=mean, sd=sd, lower_asymptote=0.05, lapse_rate=0.05, scale='linear')


class DataAnalyzer:
    """
    Class for analyzing the data from the FLE experiment.

    Attributes:
        data_dir (str): a directory containing the raw data. Must contain a 'raw' subdirectory with the raw data files,
        organized in subdirectories by participant ID.
        proj_dir (str): a directory to save the project files in, such as generated figures.
        raw_dir (str): the directory containing the raw data files.
        processed_dir (str): the directory containing the processed data files.
        fig_dir (str): the directory containing the figures.
        folders (List(str)): a list of the folders in the raw data directory, each containing the files for a participant.
        participants (str): a list of the participant IDs.
        data (dict): a dictionary containing the data for each participant, as described in README_DATA.md.
        The keys are the participant IDs, and the values are dictionaries containing the data for each participant.
        The fields of the participant dictionaries are:
            params (dict): a dictionary containing the parameters of the experiment, as read from the params.json file.
            results (pd.DataFrame): a dataframe containing the results of the experiment, as read from the results.csv file.
            training (pd.DataFrame): a dataframe containing the training results, as read from the training.csv file.
            attn_check (pd.DataFrame): a dataframe containing the attention check results, as read from the attention_checks.csv file.
            extended_data (dict): a dictionary containing the extended data for each speed. The keys are the speeds, and the
            values are xarray datasets containing the extended data for that speed.
            et_trials (pd.DataFrame): a dataframe containing the eye-tracking data for all trials.
            et_faux_trials (pd.DataFrame): a dataframe containing the eye-tracking data for all faux trials.
        excluded (dict): a dictionary containing the speeds that were excluded for each participant. The keys are the
        participant IDs, and the values are lists of speeds.
        params (pd.DataFrame): a dataframe containing the parameter estimates for each participant and speed.
        posteriors (xr.DataSet): an xarray dataset containing the posterior distributions for each participant and speed.
    """

    def __init__(self, data_dir, proj_dir):
        """
        Create a DataAnalyzer object. The object is initialized with the data directory and the project directory, and
        allows preprocessing, loading, analyzing and plotting the data.
        :param data_dir: a directory containing the raw data. Must contain a 'raw' subdirectory with the raw data files,
        organized in subdirectories by participant ID.
        :param proj_dir: a directory to save the project files in, such as generated figures.
        """

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist")
        if not os.path.isdir(proj_dir):
            raise FileNotFoundError(f"Project directory {proj_dir} does not exist")
        self.data_dir = data_dir
        self.proj_dir = proj_dir
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        if not os.path.isdir(self.raw_dir):
            raise FileNotFoundError(f"Raw data directory {self.raw_dir} does not exist")
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        if not os.path.isdir(self.processed_dir):
            os.makedirs(self.processed_dir)
        self.fig_dir = os.path.join(self.proj_dir, 'figures')
        if not os.path.isdir(self.fig_dir):
            os.makedirs(self.fig_dir)
        self._file_patterns = {
            'params': 'params.json',
            'results': 'results.csv',
            'training': 'training.csv',
            'attn_check': 'attention_checks.csv',
            'extended_data': 'extended_data',
            'et_trials': '_trials.feather',
            'et_faux_trials': '_faux.feather'
        }

        self.folders = None
        self.participants = None
        self.data = None
        self.excluded = None

        self.params = None
        self.posteriors = None

    def prepare_et_data(self, force):
        """
        Prepare the eye-tracking data for analysis. This includes converting .edf files to .asc files, and parsing the
        .asc files to .feather files.
        :param force: If true, force conversion even if an ASC file already exists (overwrites the ASC file).
        :return:
        """
        import analysis.edf as edf
        # Convert all .edf files to .asc files
        print("Converting .edf files to .asc files...")
        apply_to_dir(self.raw_dir, '.edf', edf.edf2asc, force=force)
        print("Done converting .edf files to .asc files.")
        # Parse all .asc files and save them as .feather files
        print("Parsing .asc files...")
        apply_to_dir(self.raw_dir, '.asc', edf.read_asc, save=True)
        print("Done parsing .asc files.")

    def load_data(self, excluded: Union[None, list] = None, test=False):
        """
        Load the data for all participants. The data is stored in a dictionary as `self.data`,
        with the participant ID as the key.
        :param excluded: list of participant IDs to exclude
        :param test: whether to run in test mode (only load data for the first 2 participants)
        :return:
        """

        self.folders = os.listdir(self.raw_dir)
        # Read participant ID from folder names. Each participant files are in a folder named {date}_{time}_{ID}
        participants = [f.split('_')[2] for f in self.folders]
        assert len(participants) == len(set(participants)), "There are duplicate participants"
        if test:
            participants = participants[:2]
        elif excluded is not None:
            self.folders = [f for f, p in zip(self.folders, participants) if p not in excluded]
            participants = [p for p in participants if p not in excluded]
        self.participants = participants
        self.excluded = {p: [] for p in self.participants}
        # create a dictionary to store the data for each subject
        data = {}
        print(f"Loading data for {len(self.participants)} participants...")
        for participant, folder in tzip(self.participants, self.folders, desc='Loading data'):
            data[participant] = {}
            # get all files for this participant
            participant_files = [f for f in os.listdir(os.path.join(self.raw_dir, folder))]
            for field, pattern in self._file_patterns.items():
                # get the file for this field
                file = [f for f in participant_files if pattern in f]
                # read the file
                if not file and field not in ['et_faux_trials', 'et_trials']:
                    raise FileNotFoundError(f"File {pattern} not found for participant {participant}")
                elif file:
                    data[participant][field] = DataAnalyzer._read_file(os.path.join(self.raw_dir, folder, file[0]))
        self.data = data

        param_estimates = {
            'participant': [],
            'speed': [],
            'mean_est': [],
            'mean_hdi': [],
            'sd_est': [],
            'sd_hdi': [],
            'halfpoint_est': [],
            'halfpoint_hdi': [],
            'delay_est_ms': [],
            'delay_hdi_ms': [],
        }
        posteriors = xr.Dataset()
        for participant in self.participants:
            part_ext_data = self.data[participant]['extended_data']
            speeds = self.data[participant]['params']['speeds']
            for speed in speeds:
                param_estimates['participant'].append(participant)
                param_estimates['speed'].append(speed)
                for key in param_estimates.keys():
                    if key not in ['participant', 'speed']:
                        param_estimates[key].append(self._get_final_estimate(participant, speed, key))
                posteriors[f"{participant}_{speed}"] = part_ext_data[speed]['posterior']
        self.params = pd.DataFrame.from_dict(param_estimates)
        self.posteriors = posteriors

    def print_training_results(self, silent=False, verbose=False) -> dict:
        failures = {}
        for participant in self.participants:
            if not silent:
                print(f"Participant {participant}")
            training = self.data[participant]['training']
            if (not silent) and verbose:
                for ii, row in training.iterrows():
                    print(f"Trial {ii + 1} ({row['type']}): {row['success']}")
            failures[participant] = len(training[training['success'] == False])
            if not silent:
                print(f"Failed {failures[participant]} out of {len(training)} trials")
        return failures

    def count_offsets_near_bound(self, participant, speed, bound=0.025):
        max_offset = self.data[participant]['params']['max_offset_pix']
        offset_bound = (1 - bound) * max_offset
        results = self.data[participant]['results']
        sliced_data = results.loc[(results['speed'] == speed)]
        return len(sliced_data[abs(sliced_data['intensity']) > offset_bound])

    def count_mistakes(self, participant, speed):
        """
        Count the number of mistakes for a given participant and speed. A mistake is defined as a response that does not
        match the stimulus direction, for stimuli that are outside the transition region.
        :param participant: participant ID
        :param speed: speed of the mover
        :return: number of mistakes
        """
        results = self.data[participant]['results']
        sliced_data = results.loc[(results['speed'] == speed)]
        halfpoint = self._get_final_estimate(participant, speed, 'estimated_halfpoint')
        slope = self._get_final_estimate(participant, speed, 'estimated_slope')
        trans_region = 1 / slope
        upper = halfpoint + 1 / 2 * trans_region * 1.1
        lower = halfpoint - 1 / 2 * trans_region * 1.1
        cond1 = (sliced_data['intensity'] > upper) & (sliced_data['response'] == 'left')
        cond2 = (sliced_data['intensity'] < lower) & (sliced_data['response'] == 'right')
        return len(sliced_data[cond1 | cond2])

    def percent_missing_et_data(self, participant):
        """
        Count the number of missing eye-tracking samples for a given participant.
        :param participant: participant ID
        :return: percent of missing samples
        """
        et_trials = self.data[participant]['et_trials']
        n_missing = len(et_trials) - len(et_trials.dropna())
        return n_missing / len(et_trials) * 100

    def count_for_participant(self, participant, func: callable, *args, **kwargs) -> list:
        """
        Apply a function to all speeds for a given participant

        :param participant: participant ID
        :param func: function to apply
        :param args: iterable of arguments to pass to the function
        :param kwargs: dictionary of keyword arguments to pass to the function
        :return: list of results, sorted by speed
        """
        results = []
        for speed in self.data[participant]['params']['speeds']:
            res = func(participant, speed, *args, **kwargs)
            results.append(res)
        return results

    def count_for_all_participants(self, func: callable, *args, **kwargs) -> List[List]:
        """
        Apply a function to all participants and speeds

        :param func: function to apply
        :param args: iterable of arguments to pass to the function
        :param kwargs: dictionary of keyword arguments to pass to the function
        :return: list of lists of results, sorted by participant and speed
        """
        results = []
        for participant in self.participants:
            results.append(self.count_for_participant(participant, func, *args, **kwargs))
        return results

    def plot_trial_sequence(self, participant, speed, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        sliced_data = self.data[participant]['results'].loc[self.data[participant]['results']['speed'] == speed]
        trials = sliced_data.loc[:, ['thisRepN', 'intensity', 'response', 'premature_responses',
                                     'halfpoint_est', 'sd_est']]
        trans_width_est = trials['sd_est']
        early_responses = trials[trials['premature_responses'] > 0]
        left_color = 'C0'
        right_color = 'C1'
        colors = [left_color if r == 'left' else right_color for r in trials['response']]
        ax.scatter(trials['thisRepN'], trials['intensity'], s=60, c=colors, **kwargs)
        ax.scatter(early_responses['thisRepN'], early_responses['intensity'], s=80, c='k', marker='x')
        ax.plot(trials['thisRepN'], trials['intensity'], c='k', alpha=0.5)  # add line to connect dots
        ax.plot(trials['thisRepN'], trials['halfpoint_est'], c='k', ls='--')
        upper = trials['halfpoint_est'] + 1 / 2 * trans_width_est
        lower = trials['halfpoint_est'] - 1 / 2 * trans_width_est
        ax.fill_between(trials['thisRepN'], upper, lower, color='k', alpha=0.25)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Offset [pix]')
        ax.set_title(f"Participant {participant}, speed {speed}")
        # add legend with colors according to response
        l1 = Line2D([0], [0], color=left_color, marker='o', linestyle='None')
        l2 = Line2D([0], [0], color=right_color, marker='o', linestyle='None')
        l3 = Line2D([0], [0], color='k', ls='--')
        l4 = Line2D([0], [0], color='k', marker='x', linestyle='None')
        handles = [l1, l2, l3, l4]
        labels = ['Left', 'Right', 'Halfpoint', 'Early']
        ax.legend(handles, labels)
        return ax, (handles, labels)

    def plot_rt(self, participant, speed, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        sliced_data = self.data[participant]['results']
        speed_data = sliced_data.loc[sliced_data['speed'] == speed]
        ax.scatter(speed_data['thisRepN'], speed_data['response_time_ms'], label=f"{speed}", **kwargs)
        ax.set_title(f"{speed}")
        ax.set_xlabel('Trial')
        ax.set_ylabel('RT [ms]')
        return ax, ([], [])

    def plot_for_participant(self, participant, plot_method, **kwargs):
        fig, axs = plt.subplots(2, 4, figsize=(5 * 4, 5 * 2))
        speeds = self.data[participant]['params']['speeds']
        x_label = ''
        y_label = ''
        handles = []
        labels = []
        for ii, (speed, ax) in enumerate(zip(speeds, axs.flatten())):
            ax, (handles, labels) = plot_method(participant, speed, ax=ax, **kwargs)
            if ii == 0:
                x_label = ax.get_xlabel()
                y_label = ax.get_ylabel()
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(f"speed {speed}")
        last_ax = axs.flatten()[-1]
        if len(handles) > 0:
            last_ax.legend(handles, labels, loc='center')
        last_ax.remove()
        fig.supxlabel(x_label)
        fig.supylabel(y_label)
        fig.suptitle(f"Participant {participant}")
        return fig, axs

    def plot_all_participants(self, plot_method, save_path=None, **kwargs):
        for participant in tqdm(self.participants, desc='Plotting'):
            fig, axs = self.plot_for_participant(participant, plot_method, **kwargs)
            if save_path is not None:
                try:
                    fig.savefig(os.path.join(self.fig_dir, save_path, f"{participant}.svg"))
                except FileNotFoundError:
                    os.makedirs(os.path.join(self.fig_dir, save_path))
                    fig.savefig(os.path.join(self.fig_dir, save_path, f"{participant}.svg"))

    @staticmethod
    def _read_file(file):
        file_data = None
        if file.endswith('.json'):
            with open(file, 'r') as f:
                file_data = json.load(f)
        elif file.endswith('.csv'):
            # ast.literal_eval is used to convert the string representation of a list of tuples to a list of tuples.
            # This is relevant only for results.csv, where the *_hdi are strings representing lists of tuples.
            hdi_converter = ast.literal_eval
            columns = pd.read_csv(file, nrows=0).columns.to_list()
            converters = {col: hdi_converter for col in columns if col.endswith('_hdi')}
            file_data = pd.read_csv(file, converters=converters)
            if 'results' in file:
                file_data = file_data.set_index(['thisRepN', 'speed'], drop=False)
        elif file.endswith('extended_data'):
            files = os.listdir(file)
            file_data = {}
            for f in files:
                if f.endswith('.nc'):
                    speed = int(f.split('.')[0])
                    file_data[speed] = xr.open_dataset(os.path.join(file, f))
                    assert file_data[speed].speed == speed
                else:
                    raise ValueError(f"File {f} in extended_data/ is not a .nc file.")
        elif file.endswith('.feather'):
            file_data = pd.read_feather(file)
        return file_data

    def print_attention_checks(self):
        # Print a list of the number of failed attention checks for each participant
        n_failed = np.empty(len(self.data), dtype=int)
        print("How many attention checks did each participant fail?")
        for ii, participant in enumerate(self.participants):
            attn_checks = self.data[participant]['attn_check']
            n_failed[ii] = len(attn_checks[attn_checks['success'] == False])
            print(f"Participant {participant} failed {n_failed[ii]} attention checks, "
                  f"out of {len(attn_checks)} total.")
        return n_failed

    def print_early_responses(self):
        # Print a list of the number of responses made before stimulus onset for each participant
        n_early = np.empty(len(self.data), dtype=int)
        print("How many responses were made before stimulus onset for each participant?")
        for ii, participant in enumerate(self.data):
            results = self.data[participant]['results']
            n_early[ii] = results['premature_responses'].sum()
            print(f"Participant {participant} made {n_early[ii]} responses before stimulus onset, "
                  f"out of {len(results)} total.")
        return n_early

    def plot_eye_trajectory(self, participant, trial, ax=None, **kwargs):
        """
        Plot the eye trajectory for a given participant and trial.
        :param participant: Participant ID
        :param trial: Trial number
        :param ax: axes to plot on. If none, creates a new figure and axes.
        :param kwargs: currently unused
        :return: The axes object
        """

        if ax is None:
            fig, ax = plt.subplots()
        et_data = self.data[participant]['et_trials'].loc[self.data[participant]['et_trials']['this_n'] == trial]
        response_data = self.data[participant]['results'].loc[self.data[participant]['results']['thisN'] == trial]
        flash_on = et_data[et_data['flasher']]
        flash_off = et_data[~et_data['flasher']]
        # Plot points when flasher was off
        ax.scatter(flash_off['x_pix'], flash_off['y_pix'], c=flash_off['timestamp_ms'] - et_data['timestamp_ms'].min(),
                   cmap='viridis', alpha=0.5, s=1)
        # Plot points when flasher was on
        ax.scatter(flash_on['x_pix'], flash_on['y_pix'], c=flash_on['timestamp_ms']-flash_on['timestamp_ms'].min(),
                   cmap='Reds', alpha=0.75, s=1, marker='s')
        # Add annotation for point when key was pressed
        sample_at_keypress_idx = (et_data['timestamp_ms'] - et_data['timestamp_ms'].min() -
                                  (flash_on['timestamp_ms'].min() - et_data['timestamp_ms'].min() +
                                   response_data['response_time_ms'].iloc[0])).abs().idxmin()
        sample_at_keypress = et_data.loc[sample_at_keypress_idx]
        ax.scatter(sample_at_keypress['x_pix'], sample_at_keypress['y_pix'], c='black', marker='x')

        # Format plot
        ax.set_xlabel('x [pix]')
        ax.set_ylabel('y [pix]')
        ax.set_xlim(0, self.data[participant]['params']['monitor_size_pix'][0])
        ax.set_ylim(0, self.data[participant]['params']['monitor_size_pix'][1])
        ax.set_aspect(self.data[participant]['params']['monitor_size_pix'][1] /
                      self.data[participant]['params']['monitor_size_pix'][0])
        ax.invert_yaxis()
        ax.set_title(f"Participant {participant}, trial {trial}")
        return ax, ([], [])

    def plot_posterior(self, participant, speed, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplot
        posterior = self.posteriors[f"{participant}_{speed}"].squeeze()
        if len(posterior.shape) > 2:
            warnings.warn(f"More than 2 dimensions in posterior for participant {participant} and speed {speed}. "
                          f"Marginalizing over lapse_rate and lower_asymptote.")
            posterior = posterior.mean(dim=('lapse_rate', 'lower_asymptote'))
        posterior.plot(ax=ax, rasterized=True, **kwargs)  # heatmaps are heavy as svg, so rasterize
        posterior.plot.contour(ax=ax, levels=[hdi_thresh(posterior.values)], colors='k', linewidths=0.5)
        ax.set_title(f"{speed}")
        ax.set_xscale('log')
        cbar = ax.collections[0].colorbar
        cbar.set_label('')
        return ax, ([], [])

    def plot_bitflip_posterior(self, participant, speed, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplot
        posterior_marg = self.posteriors[f"{participant}_{speed}"].mean(dim=('mean', 'sd'))
        posterior_marg.plot(ax=ax, rasterized=True, **kwargs)  # heatmaps are heavy as svg, so rasterize
        ax.set_title(f"{speed}")
        cbar = ax.collections[0].colorbar
        cbar.set_label('')
        return ax, ([], [])

    def plot_psychs_and_params(self, save_fig=False):
        data = self.data
        offset_range = np.linspace(-400, 400, 100)
        n_cols = 5
        n_rows = 3
        for ii, participant in enumerate(tqdm(data, desc='Plotting psychs and params')):
            speeds = data[participant]['params']['speeds']
            fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
            gs = mpl.gridspec.GridSpec(n_rows, n_cols, fig)
            subs = [[0, 0],  # define the subplot locations
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [1, 4],
                    [2, 4]]
            for speed, sub in zip(speeds, subs):  # plot psychometric functions overlayed with responses
                fig.add_subplot(gs[sub[0], sub[1]])
                speed_data = data[participant]['results'].loc[data[participant]['results']['speed'] == speed]
                mean = self._get_final_estimate(participant, speed, 'mean_est')
                sd = self._get_final_estimate(participant, speed, 'sd_est')
                response_after_stim = speed_data['premature_responses'] == 0
                responses = [0 if r == 'left' else 1 for
                             r in speed_data[response_after_stim]['response']]
                intensities = speed_data[response_after_stim]['intensity']
                plt.plot(offset_range, my_norm_CDF(offset_range, mean, sd).squeeze(), 'k',
                         label='Psychometric function')
                plt.scatter(intensities, responses, marker='o', label='Responses', alpha=0.25, color='C1')
                if not speeds.index(speed) == 0:
                    plt.title(f"{speed}")
                else:
                    plt.title(f"Speed: {speed}")
                    plt.xlabel(f'Offset [pix]')
                    plt.ylabel("P(\"right\")")

            cond = self.params['participant'] == participant
            # Plot estimated halfpoint vs. speed
            halfpoints = self.params[cond]['halfpoint_est']
            halfpoint_hdi = self.params[cond]['halfpoint_hdi']
            halfpoint_err = regions_to_yerr(halfpoints, halfpoint_hdi)
            ax1 = fig.add_subplot(gs[1:3, 0:2])
            plt.errorbar(speeds, halfpoints, yerr=halfpoint_err, fmt='o', label='Halfpoint', markersize=20)
            ax1.set_xscale('linear')
            plt.xlabel('Speed [pix/sec]')
            plt.ylabel('Lag [pix]')
            plt.title('Lag vs. speed')

            # Plot estimated scale vs. speed
            sds = self.params[cond]['sd_est']
            sd_hdi = self.params[cond]['sd_hdi']
            sd_err = regions_to_yerr(sds, sd_hdi)
            ax2 = fig.add_subplot(gs[1:3, 2:4])
            plt.errorbar(speeds, sds, yerr=sd_err, fmt='o', label='Slope', markersize=20)
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            plt.xlabel('Speed [pix/sec]')
            plt.ylabel(r'$\sigma$ [pix]')
            plt.title('Perceptual noise vs. speed')

            plt.suptitle(f"Responses and estimated functions, {participant}")
            if save_fig:
                subdir = 'psychometrics'
                if not os.path.exists(os.path.join(self.fig_dir, subdir)):
                    os.makedirs(os.path.join(self.fig_dir, subdir))
                path = os.path.join(self.fig_dir, subdir, f'psychs_{participant}.svg')
                plt.savefig(path)

    def plot_halfpoints(self, save_fig=False, lin_fit=False):
        # Plot the halfpoint vs. speed for each participant, with uncertainty as error bars
        data = self.data
        fig, axes = tile_subplots(len(data))
        for ax, participant in zip(axes, data):
            speeds = data[participant]['params']['speeds']
            halfpoints = np.empty(len(speeds))
            hdis = []
            errs = np.empty((len(speeds)))
            for ss, speed in enumerate(speeds):
                halfpoints[ss] = self._get_final_estimate(participant, speed, 'halfpoint_est')
                hdis.append(self._get_final_estimate(participant, speed, 'halfpoint_hdi'))
                errs[ss] = self._get_error(participant, speed, 'halfpoint_hdi')
            halfpoint_errors = regions_to_yerr(halfpoints, hdis)
            ax.errorbar(speeds, halfpoints, yerr=halfpoint_errors, marker='o')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_title(f"{participant}")
            ax.set_ylim([-100, 225])

            if lin_fit:
                coefs = poly.polyfit(speeds, halfpoints, 1, w=1 / errs)
                ax.plot(speeds, poly.polyval(speeds, coefs), 'k--', label=f'm={coefs[-1] * 1000:.2f} ms')
                ax.legend()
        fig.suptitle("Estimated lag vs. speed for all participants")
        fig.supxlabel("Speed [pix/sec]")
        fig.supylabel("Lag [pix]")
        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(self.fig_dir, 'halfpoints_vs_speed.svg'))

    def plot_widths(self, save_fig=False, lin_fit=False):
        # plot the slope vs. speed for each participant, with uncertainty as error bars
        data = self.data
        fig, axes = tile_subplots(len(data))
        for ax, participant in zip(axes, data):
            speeds = data[participant]['params']['speeds']
            widths = np.empty(len(speeds))
            hdis = []
            errs = np.empty((len(speeds)))
            for ss, speed in enumerate(speeds):
                widths[ss] = self._get_final_estimate(participant, speed, 'sd_est')
                hdis.append(self._get_final_estimate(participant, speed, 'sd_hdi'))
                errs[ss] = self._get_error(participant, speed, 'sd_hdi')
            slope_errors = regions_to_yerr(widths, hdis)
            ax.errorbar(speeds, widths, yerr=slope_errors, marker='o')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_title(f"{participant}")
            ax.set_ylim([5, 500])
            ax.set_yscale('log')
            ax.set_xscale('log')

            if lin_fit:
                coefs = poly.polyfit(np.log(speeds), np.log(widths), 1, w=1 / errs)
                pred = np.exp(coefs[0]) * speeds ** coefs[1]
                ax.plot(speeds, pred, 'k--', label=f'm={coefs[-1]:.2f}')
                ax.legend()
        fig.suptitle("Width vs. speed")
        fig.supxlabel("Speed [pix/sec]")
        fig.supylabel(r"$\sigma$ [pix]")
        if save_fig:
            plt.savefig(os.path.join(self.fig_dir, 'widths_vs_speed.svg'))

    def plot_average_halfpoints(self, save_fig=False, lin_fit=False):
        # plot the average halfpoint vs. speed for all participants
        data = self.data
        speeds = data[list(data.keys())[0]]['params']['speeds']  # assumes all participants have same speeds
        halfpoints = np.empty(len(speeds))
        halfpoint_errors = np.empty(len(speeds))
        for speed in speeds:
            this_speed_halfpoints = np.empty(len(data))
            halfpoint_uncertainties = np.empty(len(data))
            for ii, participant in enumerate(data):
                this_speed_halfpoints[ii] = self._get_final_estimate(participant, speed, 'halfpoint_est')
                halfpoint_uncertainties[ii] = self._get_error(participant, speed, 'halfpoint_hdi')
            halfpoints[speeds.index(speed)] = np.average(this_speed_halfpoints, weights=1 / halfpoint_uncertainties)
            halfpoint_errors[speeds.index(speed)] = np.sqrt(
                np.cov(this_speed_halfpoints, aweights=1 / halfpoint_uncertainties))
            # TODO take into account excluded participants
        plt.figure()
        plt.scatter(self.params['speed'], self.params['halfpoint_est'], c='C1', alpha=0.2, zorder=10,
                    label='Individual lag')
        plt.errorbar(speeds, halfpoints, yerr=halfpoint_errors, marker='o', c='C0', label='Average lag')
        if lin_fit:
            coefs = poly.polyfit(speeds, halfpoints, 1, w=1 / halfpoint_errors)
            plt.plot(speeds, poly.polyval(speeds, coefs), 'k--', zorder=0,
                     label='Linear fit, m = {:.2f} ms'.format(coefs[-1] * 1000))
        plt.xlabel("Speed [pix/sec]")
        plt.ylabel("Lag [pix]")
        plt.title("Average halfpoint vs. speed")
        plt.legend()
        if save_fig:
            plt.savefig(os.path.join(self.fig_dir, 'average_halfpoints_vs_speed.svg'))

    def plot_average_widths(self, save_fig=False, lin_fit=False):
        # plot the average slope vs. speed for all participants
        data = self.data
        speeds = data[list(data.keys())[0]]['params']['speeds']
        sds = np.empty(len(speeds))
        sd_errors = np.empty(len(speeds))
        for speed in speeds:
            this_speed_sds = np.empty(len(data))
            sd_uncertainties = np.empty(len(data))
            for ii, participant in enumerate(data):
                this_speed_sds[ii] = self._get_final_estimate(participant, speed, 'sd_est')
                sd_uncertainties[ii] = self._get_error(participant, speed, 'sd_hdi')
            sds[speeds.index(speed)] = np.average(this_speed_sds, weights=1 / sd_uncertainties)
            sd_errors[speeds.index(speed)] = np.sqrt(np.cov(this_speed_sds, aweights=1 / sd_uncertainties))
            # TODO take into account excluded participants
        individual_data = self.params['sd_est']
        data_label = r'$\sigma$'
        plt.figure()
        plt.scatter(self.params['speed'], individual_data, c='C1', alpha=0.2, zorder=10,
                    label='Individual ' + data_label)
        plt.errorbar(speeds, sds, yerr=sd_errors, marker='o', c='C0', label='Average ' + data_label)
        if lin_fit:
            coefs = poly.polyfit(np.log(speeds), np.log(sds), 1, w=1 / sd_errors)
            pred = np.exp(coefs[0]) * speeds ** coefs[1]  # this is needed because the fit is done in log space
            plt.plot(speeds, pred, 'k--', zorder=0, label='Linear fit, m = {:.2f}'.format(coefs[-1]))
        plt.xlabel("Speed [pix/sec]")
        plt.ylabel(r"$\sigma$ [pix]")
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"Average {data_label} vs. speed")
        plt.legend()
        if save_fig:
            plt.savefig(os.path.join(self.fig_dir, 'average_widths_vs_speed.svg'))

    @staticmethod
    def _linear_transform(vector, a, b):
        min_value = np.min(vector)
        max_value = np.max(vector)
        transformed_vector = ((vector - min_value) / (max_value - min_value)) * (b - a) + a
        return transformed_vector

    def _get_error(self, participant, speed, param):
        assert 'hdi' in param, 'param must be an HDI'
        hdi_ = self._get_final_estimate(participant, speed, param)
        if len(hdi_) == 1:
            error = hdi_[0][1] - hdi_[0][0]
        else:
            raise ValueError(f"{param} HDI has more than one region")
        return error

    def _get_final_estimate(self, participant, speed, param):
        if self.params is None:
            this_participant_data = self.data[participant]['results']
            this_speed_data = this_participant_data.loc[this_participant_data['speed'] == speed]
            max_rep = this_speed_data['thisRepN'].max()
            res = this_speed_data.loc[this_speed_data['thisRepN'] == max_rep, param].item()
        else:
            cond1 = (self.params['participant'] == participant)
            cond2 = (self.params['speed'] == speed)
            res = self.params[cond1 & cond2][param].item()
        return res

    def calc_v(self, window=39):
        """
        Preprocess eye tracking data. Add columns for speed in pix/ms.
        """
        # TODO: add conversion to visual angle
        def calc_v(df):
            df = df.sort_values('timestamp_ms')
            x_filtered = df['x_pix'].rolling(window=window, center=True).mean()
            y_filtered = df['y_pix'].rolling(window=window, center=True).mean()
            df['v_x'] = x_filtered.diff() / df['timestamp_ms'].diff() * 1000
            df['v_y'] = y_filtered.diff() / df['timestamp_ms'].diff() * 1000
            return df[['v_x', 'v_y']]

        data = self.data
        for participant in data:
            cols = ['et_trials', 'et_faux_trials']
            for col in cols:
                dat = data[participant][col]
                v_df = dat.groupby('this_n').apply(calc_v)
                dat['v_x'] = v_df['v_x']
                dat['v_y'] = v_df['v_y']
                data[participant][col] = dat

    def reflect_trials(self):
        """
        Reflect the x position of the eye tracking data for trials where the motion was from left to right.
        :return:
        """
        data = self.data
        # reflect the x position if the motion was from left to right
        for participant in data:
            results = data[participant]['results']
            screen_center = data[participant]['params']['monitor_size_pix'][0] / 2
            et_trials = data[participant]['et_trials']
            et_trials['x_pix_ltr_reflect'] = et_trials['x_pix']
            for trial_id in results['thisN']:
                ltr = results.loc[results['thisN'] == trial_id, 'motion_left_to_right'].iloc[0]
                if not ltr:
                    et_trials.loc[et_trials['this_n'] == trial_id, 'x_pix_ltr_reflect'] = reflect(et_trials['x_pix'],
                                                                                                  screen_center)
            data[participant]['et_trials'] = et_trials
        self.data = data

    def trial_time(self, participant, trial_id):
        """
        Calculate the time in ms of the trial, relative to the start of the trial.
        :param participant: participant ID
        :param trial_id: Number of the trial
        :return: time in ms
        """
        trial_data = self.data[participant]['et_trials'].loc[self.data[participant]['et_trials']['this_n'] == trial_id]
        return trial_data['timestamp_ms'] - trial_data['timestamp_ms'].min()

    def participant_mover_x_t(self, participant, trial_id, t):
        """
        Calculate the position of the moving object at time t for a given participant and trial.
        :param participant: participant ID
        :param trial_id: Number of the trial
        :param t: time
        """

        trial_data = self.data[participant]['results'].loc[self.data[participant]['results']['thisN'] == trial_id]
        speed = trial_data['speed'].iloc[0]
        dir = 1 if trial_data['motion_left_to_right'].iloc[0] else -1
        speed = speed * dir
        motion_span = self.data[participant]['params']['motion_span_pix']
        x0 = self.data[participant]['params']['monitor_size_pix'][0] / 2 - dir * motion_span / 2
        return constant1d(t, x0, speed)

    def residual(self, participant, trial_id):
        """
        Calculate the residual between the eye position and the moving object position for a given participant and trial.
        :param participant: participant ID
        :param trial_id: Number of the trial
        """

        et_data = self.data[participant]['et_trials'].loc[self.data[participant]['et_trials']['this_n'] == trial_id]
        t = et_data['timestamp_ms'] - et_data['timestamp_ms'].min()
        x = et_data['x_pix']
        x_mover = self.participant_mover_x_t(participant, trial_id, t / 1000)
        residuals = x_mover - x
        return residuals

    def is_pursuit(self, participant, trial_id, threshold=0.4):
        """
        Return whether the eye movement in the trial is consistent with pursuit.
        :param participant: participant ID
        :param trial_id: Number of the trial
        :param threshold: threshold for the relative mean deviation from the mover position (max 0.5)
        :return: True if the eye movement is consistent with pursuit, False otherwise
        """

        if threshold > 0.5 or threshold < 0:
            raise ValueError("Threshold must be between 0 and 0.5")
        residuals = self.residual(participant, trial_id)
        thresh = threshold * abs(residuals).max()
        return residuals.abs().mean() < thresh

    def pursuit_onset(self, et_data=None, participant=None, trial_id=None, window=21):
        """
        Find the onset of pursuit in a trial.
        :param et_data: eye tracking data for the trial. If None, participant and trial_id must be provided to retrieve
        the data from self.data.
        :param participant: participant ID. Must be provided if et_data is None.
        :param trial_id: number of the trial. Must be provided if et_data is None.
        :param window: window size for smoothing the eye position. Must be odd. If None or 1, no smoothing is done.
        :return: time in ms of the onset of pursuit, relative to trial start time
        """

        # validate input - if trial_data is not None, participant and trial_id must be None, and vice versa.
        if et_data is not None:
            if participant is not None or trial_id is not None:
                raise ValueError("If trial_data is not None, participant and trial_id must be None.")
            ltr = None
        else:
            if participant is None or trial_id is None:
                raise ValueError("If trial_data is None, participant and trial_id must not be None.")
            if not self.is_pursuit(participant, trial_id):
                return np.nan
            et_data = self.data[participant]['et_trials'].loc[self.data[participant]['et_trials']['this_n'] ==
                                                              trial_id]
            ltr = self.data[participant]['results'].loc[self.data[participant]['results']['thisN'] ==
                                                        trial_id, 'motion_left_to_right'].iloc[0]

        ONSET_START = 100
        ONSET_END = 500

        if window is None:
            window = 1
        t = et_data['timestamp_ms'] - et_data['timestamp_ms'].min()
        x = et_data['x_pix']
        x_filtered = x.rolling(window=window, center=True).mean()
        x_trimmed = x_filtered.loc[t.between(ONSET_START, ONSET_END)]
        t_trimmed = t.loc[t.between(ONSET_START, ONSET_END)]
        t_onset = pursuit_onset(x_trimmed, t_trimmed, ltr)
        return t_onset + ONSET_START

    def plot_et_residuals(self, participant, speed, ltr, only_pursuit, align=False, ax=None, **kwargs):
        """
        Plot the residuals between the eye position and the moving object position for a given participant, speed and
        direction. If only_pursuit is True, only trials consistent with pursuit will be plotted. Plot the mean residual
        and the standard deviation as a shaded area.
        :param participant: participant ID
        :param speed: mover speed
        :param ltr: Direction of motion (left to right)
        :param only_pursuit: whether to include only pursuit trials
        :param align: whether to align the trials to the onset of pursuit
        :param ax: axes to plot on. If none, creates a new figure and axes.
        :param kwargs:
        :return: axes object
        """

        MAX_ONSET = 500
        if ax is None:
            fig, ax = plt.subplots()
        individual_styling = {'linewidth': 0.75, 'alpha': 0.3}
        trials = self.data[participant]['results']
        trials = trials[(trials['speed'] == speed) & (trials['motion_left_to_right'] == ltr)]
        et_trials = self.data[participant]['et_trials']
        # filter out trials with no pursuit
        if only_pursuit:
            trials = trials[trials['thisN'].apply(lambda x: self.is_pursuit(participant, x))]
        max_len = max([t['timestamp_ms'].max() - t['timestamp_ms'].min() for _, t in et_trials.groupby('this_n')]) + 1
        # create an array to store the residuals, accounting for alignment
        residuals = np.full((len(trials), MAX_ONSET + max_len), np.nan)
        # Plot individual trials, coloring differently the times with flasher on
        for ii, trial_id in enumerate(trials['thisN']):
            et_trial = et_trials.loc[et_trials['this_n'] == trial_id]
            residual = self.residual(participant, trial_id)
            zeroed_time = et_trial['timestamp_ms'] - et_trial['timestamp_ms'].min()
            flash_times = et_trial[et_trial['flasher']]['timestamp_ms'] - et_trial['timestamp_ms'].min()
            onset = 0
            if align:
                onset = self.pursuit_onset(participant=participant, trial_id=trial_id)
            if np.isnan(onset):
                continue
            zeroed_time -= onset
            flash_times -= onset
            ax.plot(zeroed_time, residual, color='C0', **individual_styling)
            flash_on = et_trial[et_trial['flasher']]
            ax.plot(flash_times, residual[flash_on.index], color='C2', **individual_styling)
            residuals[ii, MAX_ONSET - onset:len(residual) + MAX_ONSET - onset] = residual
        # Plot mean and standard deviation
        mean_residual = np.nanmean(residuals, axis=0)
        std_residual = np.nanstd(residuals, axis=0)
        ax.plot(range(-MAX_ONSET, max_len), mean_residual, color='C1')
        ax.fill_between(range(-MAX_ONSET, max_len), mean_residual - std_residual, mean_residual + std_residual,
                        color='C1', alpha=0.25)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Residual [pix]')
        ax.set_title(f"Participant {participant}, speed {speed}, {'LTR' if ltr else 'RTL'}")
        return ax, ([], [])

    def reestimate_params(self, filename='reestimated'):
        """
        Re-estimate the psychometric function parameters for all participants and speeds, ignoring premature responses.
        :param filename: The filename to save the re-estimated parameters to. Can include a path.
        :param full: Whether to re-estimate all participants and conditions, regardless of whether they have premature
        responses. If False, only participants with premature responses will be re-estimated.
        :return:
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        data = self.data
        scale = 'linear'
        new_param_estimates = {
            'participant': [],
            'speed': [],
            'mean_est': [],
            'mean_hdi': [],
            'sd_est': [],
            'sd_hdi': [],
            'halfpoint_est': [],
            'halfpoint_hdi': [],
            'delay_est_ms': [],
            'delay_hdi_ms': [],
        }
        posteriors_dataset = xr.Dataset()
        print("Re-estimating psychometric function parameters for all participants and speeds...")
        for participant in tqdm(data, desc='Participants', position=0):
            participant_data = data[participant]['results']
            speeds = data[participant]['params']['speeds']
            for speed in tqdm(speeds, desc='Speeds', position=1, leave=False):
                # get the data for this speed condition
                cond_data = participant_data.loc[participant_data['speed'] == speed]
                intensities = cond_data['intensity']
                extended_data = data[participant]['extended_data'][speed]
                prior = extended_data['prior']
                qp = QuestPlusNorm(
                    intensities=np.sort(intensities.unique()),
                    means=prior['mean'],
                    sds=prior['sd'],
                    lower_asymptotes=[0.05],
                    lapse_rates=[0.05],
                    responses=['right', 'left'],
                    stim_scale=scale,
                    param_estimation_method='mean'
                )
                for datum in cond_data.itertuples():
                    qp.update(response=datum.response, intensity=datum.intensity)

                # build dictionary of new parameter estimates
                new_param_estimates['participant'].append(participant)
                new_param_estimates['speed'].append(speed)

                # mean
                mean_est = qp.param_estimate['mean']
                mean_hdi = hdi(qp.marginal_posterior['mean'], qp.param_domain['mean'])
                new_param_estimates['mean_est'].append(mean_est)
                new_param_estimates['mean_hdi'].append(mean_hdi)

                # sd
                sd_est = qp.param_estimate['sd']
                sd_hdi = hdi(qp.marginal_posterior['sd'], qp.param_domain['sd'])
                new_param_estimates['sd_est'].append(sd_est)
                new_param_estimates['sd_hdi'].append(sd_hdi)

                # halfpoint (same as mean - legacy)
                halfpoint_est = mean_est
                halfpoint_hdi = mean_hdi
                new_param_estimates['halfpoint_est'].append(halfpoint_est)
                new_param_estimates['halfpoint_hdi'].append(halfpoint_hdi)

                # lag
                delay_est_ms = halfpoint_est / speed * 1000
                delay_hdi_ms = hdi(qp.marginal_posterior['mean'], qp.param_domain['mean'] / speed * 1000)
                new_param_estimates['delay_est_ms'].append(delay_est_ms)
                new_param_estimates['delay_hdi_ms'].append(delay_hdi_ms)

                posteriors_dataset[f"{participant}_{speed}"] = qp.posterior
        df = pd.DataFrame.from_dict(new_param_estimates)
        df.to_csv(os.path.join(self.processed_dir, f'{filename}_params.csv'))
        posteriors_dataset.to_netcdf(os.path.join(self.processed_dir, f'{filename}_posteriors.nc'), 'w',
                                     format='NETCDF4')
        warnings.filterwarnings("default", category=RuntimeWarning)

    def load_reestimated_params(self, filename='reestimated'):
        """
        Load re-estimated parameters from file.
        :param filename: The filename of the re-estimated parameters and posteriors. Can be a path in the processed
        directory.
        """
        params = pd.read_csv(os.path.join(self.processed_dir, f'{filename}_params.csv'), dtype={'participant': str})
        posteriors = xr.open_dataset(os.path.join(self.processed_dir, f'{filename}_posteriors.nc'))
        self.params = params
        self.posteriors = posteriors
        return params, posteriors

    def calc_response_rates(self):
        """Calculate response rates for each participant and speed."""
        response_rates = {}
        for participant in self.participants:
            response_rates[participant] = {}
            participant_data = self.data[participant]['results']
            participant_data = participant_data[participant_data['response_before_stim'] == False]
            for speed in self.data[participant]['params']['speeds']:
                speed_data = participant_data[participant_data['speed'] == speed]
                responses = speed_data['response'].values
                responses = np.asarray([r == 'right' for r in responses])
                response_rates[participant][speed] = np.mean(responses)
        return response_rates

    def estimate_delay_sec(self, participant) -> float:
        speeds = self.data[participant]['params']['speeds']
        halfpoints = np.zeros(len(speeds))
        halfpoint_errors = np.zeros(len(speeds))
        for i, speed in enumerate(speeds):
            halfpoints[i] = self._get_final_estimate(participant, speed, 'mean_est')
            halfpoint_errors[i] = self._get_error(participant, speed, 'mean_hdi')
        coeffs = poly.polyfit(speeds, halfpoints, 1, w=1 / halfpoint_errors)
        return coeffs[1]

    def estimate_K(self, participant) -> float:
        tau = self.estimate_delay_sec(participant)

        def sd_of_v(v, K, A, B):
            return np.sqrt(1 / (2 * K) * (A + tau ** 2 * B / 2 * v))

        speeds = self.data[participant]['params']['speeds']
        sds = np.zeros(len(speeds))
        sd_errors = np.zeros(len(speeds))
        for i, speed in enumerate(speeds):
            sds[i] = self._get_final_estimate(participant, speed, 'sd_est')
            sd_errors[i] = self._get_error(participant, speed, 'sd_hdi')
        popt = curve_fit(sd_of_v, speeds, sds, sigma=sd_errors / 2, absolute_sigma=True, bounds=(0, np.inf))[0]
        K_est = popt[0]
        return K_est
