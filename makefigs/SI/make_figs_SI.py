"""
make_figs_SI.py
Generate figures for the SI.

This script generates the figures for the SI of the paper. It uses the data from the Vanilla and ET experiments.
It relies on preprocessed data - see the usage section in the `README` file for details on how to preprocess the data.

Usage:
    python make_figs_SI.py <data_path> [--output_path <output_path>]
Arguments:
    data_path: Path to the directory containing the data files.
    --output_path: Path to save the figures. Defaults to the current directory.

Example:
    python make_figs_SI.py /path/to/data --output_path /path/to/save
"""

from argparse import ArgumentParser
import itertools

from matplotlib.pyplot import subplots

import makefigs.common as c
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from ComDePy.viz import set_defaults
import xarray as xr
import pandas as pd

from analysis.gaussian_process import marginalize_lse
from analysis.scripts.make_constants_tex import format_value_with_error
from makefigs.main.make_figs_main import (load_bfc_vanilla, load_mle, load_bfc_et, exclude_bfc,
                                          pearsonr_CI, get_latest_estimate, mover_speed_label)
from analysis.DataAnalyzer import my_norm_CDF
from analysis.scripts.calc_mle_from_llk import load_llks
from util.FLE_utils import regions_to_yerr, hdi_thresh
from util.xarray import bootstrap, coordmax


def set_local_defaults():
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.labelweight'] = 'normal'


def tile_subplots(n, **kwargs):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = subplots(rows, cols, **kwargs)
    # remove empty axes
    for i in range(n, rows * cols):
        fig.delaxes(axes.flatten()[i])
    return fig, axes


def load_bfc_llks(path):
    data = {}
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.nc'):
                prob = xr.load_dataset(os.path.join(root, file))
                data.setdefault(prob.attrs['long_name'][:8], {})[prob.attrs['speed']] = prob['posterior']
    return data


def make_fidelity_regressions_fig(mle_data):
    def prepare_data(data):
        data = data.copy()
        data['fidelity'] = data['K'] / (data['K'] + data['b'])
        return data

    def plot(data):
        data_grouped = data.groupby('part_id')
        # TODO add error bars
        fig, axs = tile_subplots(len(data_grouped), figsize=(16, 16), sharex=True, sharey=True)
        for i, (part_id, part_data) in enumerate(data_grouped):
            ax = axs.flatten()[i]
            part_data = part_data.reset_index()
            rho, ci, p = pearsonr_CI(part_data['speed'], part_data['fidelity'])
            sns.regplot(x='speed', y='fidelity', data=part_data, ax=ax,
                        label=f'ρ = {rho:.1f} [{ci[0]:.2f}, {ci[1]:.2f}]\np = {p:.2e}')
            ax.axhline(1, color='k', linestyle='--', alpha=0.5)
            ax.legend(fontsize='smaller')
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(0, 1600)
            ax.set_ylim(0.45, 1.05)

        fig.supxlabel(c.LABELS['mover speed name'] + ' ' + c.LABELS['speed units'])
        fig.supylabel(c.LABELS['fidelity name'] + ' ' + c.LABELS['fidelity symbol'])
        return fig

    mle_data = prepare_data(mle_data)
    fig = plot(mle_data)
    return fig


def make_lag_vs_speed_fig(bfc_data):
    def prepare_data(data):
        return get_latest_estimate(data)

    def plot(data):
        n_parts_per_fig = 30
        data_grouped = data.groupby('part_id')
        n_figs = int(np.ceil(len(data_grouped) / n_parts_per_fig))
        figs = []
        for i in range(n_figs):
            n_panels = min(n_parts_per_fig, len(data_grouped) - i * n_parts_per_fig)
            fig, axs = tile_subplots(n_panels, figsize=(16, 16), sharex=True, sharey=True)
            axs = axs.flatten()
            for jj, (part_id, part_data) in enumerate(itertools.islice(data_grouped, i * n_parts_per_fig, None)):
                ax = axs[jj]
                _, _, p = pearsonr_CI(part_data.reset_index()['speed'], part_data['halfpoint_est'])
                slope = np.polyfit(part_data.reset_index()['speed'], part_data['halfpoint_est'], 1)[0] * 1000
                slope_std = np.polyfit(part_data.reset_index()['speed'], part_data['halfpoint_est'], 1, cov=True)[1][0, 0] * 1000
                slope_with_err = format_value_with_error(slope, slope_std, limit=2)
                label = f'slope = ${slope_with_err}$\np = {p:.2e}'
                sns.regplot(x='speed', y='halfpoint_est', data=part_data.reset_index(), ax=ax, label=label)
                try:
                    errs = regions_to_yerr(part_data['halfpoint_est'].to_list(), part_data['halfpoint_hdi'].to_list())
                    ax.errorbar(part_data.reset_index()['speed'], part_data['halfpoint_est'], yerr=errs, fmt='none', capsize=5)
                except AssertionError or ValueError:
                    ax.plot(part_data.reset_index()['speed'], part_data['halfpoint_est'], 'o', color='C0')
                ax.set_title('')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim(0, 1600)
                ax.set_ylim(-300, 300)
                ax.grid(False)
                ax.legend(fontsize='x-small')
                if jj >= n_panels - 1:
                    break
            fig.supxlabel(mover_speed_label)
            fig.supylabel(c.LABELS['lag name'] + ' ' + c.LABELS['location units'])
            figs.append(fig)
        return figs

    bfc_data = prepare_data(bfc_data)
    figs = plot(bfc_data)
    return figs


def make_bfc_noise_fig(bfc_data):
    def prepare_data(data):
        return get_latest_estimate(data)

    def plot(data):
        n_parts_per_fig = 30
        data_grouped = data.groupby('part_id')
        n_figs = int(np.ceil(len(data_grouped) / n_parts_per_fig))
        figs = []
        for i in range(n_figs):
            n_panels = min(n_parts_per_fig, len(data_grouped) - i * n_parts_per_fig)
            fig, axs = tile_subplots(n_panels, figsize=(16, 16), sharex=True, sharey=True)
            axs = axs.flatten()
            for jj, (part_id, part_data) in enumerate(itertools.islice(data_grouped, i * n_parts_per_fig, None)):
                ax = axs[jj]
                x = np.log10(part_data.reset_index()['speed'])
                y = np.log10(part_data['sd_est'])
                _, _, p = pearsonr_CI(x, y)
                slope = np.polyfit(x, y, 1)[0]
                slope_std = np.polyfit(x, y, 1, cov=True)[1][0, 0]
                slope_with_err = format_value_with_error(slope, slope_std, limit=2)
                label = f'slope = ${slope_with_err}$\np = {p:.2e}'
                sns.regplot(x=x, y=y, ax=ax, label=label)
                try:
                    errs = regions_to_yerr(y.to_list(), np.log10(part_data['sd_hdi'].to_list()))
                    ax.errorbar(x, y, yerr=errs, fmt='none', capsize=5)
                except ValueError:
                    ax.plot(x, y, 'o', color='C0')
                ax.set_title('')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.grid(False)
                ax.legend(fontsize='x-small')
                if jj >= n_panels - 1:
                    break
            fig.supxlabel(mover_speed_label)
            fig.supylabel(" ".join((c.LABELS['perceptual noise name'],
                                    c.LABELS['perceptual noise symbol'],
                                    c.LABELS['location units'])))
            figs.append(fig)
        return figs

    data = prepare_data(bfc_data)
    figs = plot(data)
    return figs


def make_psychometric_curves_fig(bfc_data):
    def plot(data):
        n_participants_to_show = 7
        fig, axs = plt.subplots(nrows=n_participants_to_show, ncols=7, figsize=(16, 16), sharex=True, sharey=True)
        for i, (part_id, part_data) in enumerate(data.groupby('part_id')):
            for j, (speed, speed_data) in enumerate(part_data.groupby('speed')):
                ax = axs[i, j]
                y = [0 if r == 'left' else 1 for r in speed_data['response']]
                ax.scatter(speed_data['offset'], y, alpha=0.15, s=30)

                final_estimate = get_latest_estimate(speed_data)
                mean = final_estimate['halfpoint_est']
                sd = final_estimate['sd_est']
                x = np.linspace(-400, 400, 100)
                ax.plot(x, my_norm_CDF(x, mean, sd).squeeze(), '-k')

                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xlim(-400, 400)
                ax.set_ylim(-0.05, 1.05)
                ax.grid(False)
                if i == 0:
                    ax.set_title(f'{int(speed)}')
            if i >= n_participants_to_show - 1:
                break
        fig.supxlabel('Offset ' + c.LABELS['location units'])
        fig.supylabel('Probability responding \'right\', trial responses')
        return fig

    fig = plot(bfc_data)
    return fig


def make_2d_likelihood_fig(llk_data):
    def plot(data):
        n_parts_per_fig = 10
        n_figs = int(np.ceil(len(data) / n_parts_per_fig))
        figs = []
        for i in range(n_figs):
            nrows = min(n_parts_per_fig, len(data) - i * n_parts_per_fig)
            fig, axs = plt.subplots(nrows=nrows, ncols=7, figsize=(16, 16), sharex=True, sharey=True)
            for ii, (part_id, part_data) in enumerate(itertools.islice(data.items(), i * n_parts_per_fig, None)):
                # sort by speed
                part_data = {speed: llk for speed, llk in sorted(part_data.items(), key=lambda x: x[0])}
                for jj, (speed, llk) in enumerate(part_data.items()):
                    if axs.ndim == 1:
                        ax = axs[jj]
                    else:
                        ax = axs[ii, jj]
                    marginal = marginalize_lse(llk.sum('epoch'), ['b', 'sig_y'])
                    # plot the 2d likelihood in ax log scale
                    _ = ax.pcolormesh(marginal['K'], marginal['sig_r'], marginal.values,
                                          shading='auto')

                    percentiles = [70, 90, 99.5]
                    c_levels = [np.percentile(marginal.values, p) for p in percentiles]
                    _ = ax.contour(marginal['K'], marginal['sig_r'], marginal.values,
                                   levels=c_levels,
                                   colors='k',
                                   linewidths=1)

                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_title('')
                    ax.grid(False)
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    if ii == 0:
                        ax.set_title(f'{int(speed)}')
                if ii >= nrows - 1:
                    break
            fig.supylabel(' '.join((c.LABELS['K name'], c.LABELS['K symbol'], c.LABELS['K units'])))
            fig.supxlabel(' '.join((c.LABELS['et noise name'], c.LABELS['et noise symbol'], c.LABELS['location units'])))
            figs.append(fig)
        return figs

    figs = plot(llk_data)
    return figs


def make_1d_likelihood_fig(llk_data):
    def plot(data):
        n_parts_per_fig = 10
        n_figs = int(np.ceil(len(data) / n_parts_per_fig))
        figs = []
        for i in range(n_figs):
            nrows = min(n_parts_per_fig, len(data) - i * n_parts_per_fig)
            fig, axs = plt.subplots(nrows=nrows, ncols=7, figsize=(16, 16), sharex=True)
            for ii, (part_id, part_data) in enumerate(itertools.islice(data.items(), i * n_parts_per_fig, None)):
                # sort by speed
                part_data = {speed: llk for speed, llk in sorted(part_data.items(), key=lambda x: x[0])}
                for jj, (speed, llk) in enumerate(part_data.items()):
                    if axs.ndim == 1:
                        ax = axs[jj]
                    else:
                        ax = axs[ii, jj]
                    marginal = marginalize_lse(llk.sum('epoch'), ['K', 'sig_r', 'sig_y'])
                    marginal.plot.line(x='b', ax=ax, marker='o')
                    ax.set_xlabel('')
                    ax.set_ylabel('')
                    ax.set_title('')
                    ax.set_xscale('log')
                    ax.grid(False)
                    if ii == 0:
                        ax.set_title(f'{int(speed)}')
                if ii >= nrows - 1:
                    break
            fig.supxlabel(' '.join((c.LABELS['b name'], c.LABELS['b symbol'], c.LABELS['b units'])))
            fig.supylabel('Marginal log likelihood')
            figs.append(fig)
        return figs

    figs = plot(llk_data)
    return figs


def make_individual_K_fig(mle, llk):
    def prepare_data(llk, mle):
        stds = []
        for part_id, part_data in llk.items():
            for speed, speed_data in part_data.items():
                bs = bootstrap(marginalize_lse(speed_data, ['sig_y', 'sig_r', 'b']), sample_dim='epoch',
                               statistic='sum', dim='epoch')
                bootstrapped_ks = []
                for jj in range(len(bs['resample'])):
                    resample = bs.sel(resample=jj)
                    mle_K = coordmax(resample)['K']
                    bootstrapped_ks.append(mle_K)
                std = np.std(bootstrapped_ks)
                stds.append({'part_id': part_id, 'speed': speed, 'K_std': std})
        df = pd.DataFrame(stds, columns=['part_id', 'speed', 'K_std'])
        mle = mle.copy()
        mle = mle.reset_index()
        df = df.set_index(['part_id', 'speed'])
        mle = pd.merge(mle, df, on=['part_id', 'speed'], how='left')
        return mle

    def plot(mle):
        mle_grouped = mle.groupby('part_id')
        fig, axs = tile_subplots(len(mle_grouped), figsize=(16, 16), sharex=True, sharey=True)
        for i, (part_id, part_data) in enumerate(mle_grouped):
            # sort by speed
            part_data = part_data.sort_values(by='speed')
            ax = axs.flatten()[i]
            ax.errorbar(part_data['speed'], part_data['K'], yerr=part_data['K_std'])
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yscale('log')
            ax.set_xlim(0, 1600)
        fig.supylabel(' '.join((c.LABELS['K name'], c.LABELS['K symbol'], c.LABELS['K units'])))
        fig.supxlabel(' '.join((c.LABELS['mover speed name'], c.LABELS['mover speed symbol'], c.LABELS['speed units'])))
        return fig

    data = prepare_data(llk, mle)
    fig = plot(data)
    return fig


def make_psych_2d_posterior_fig(prob_data):
    n_participants_to_show = 7
    fig, axs = plt.subplots(nrows=n_participants_to_show, ncols=7, figsize=(16, 16), sharex=True, sharey=True)
    for i, (part_id, part_data) in enumerate(prob_data.items()):
        part_data = {speed: post for speed, post in sorted(part_data.items(), key=lambda x: x[0])}
        for j, (speed, post) in enumerate(part_data.items()):
            ax = axs[i, j]
            post = post.mean(dim=['lapse_rate', 'lower_asymptote'])
            _ = ax.pcolormesh(post['mean'], post['sd'], post.values, shading='auto')
            # ax.set_xscale('log')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            ax.set_ylim(0, 100)
            ax.grid(False)
            if i == 0:
                ax.set_title(f'{int(speed)}')
        if i >= n_participants_to_show - 1:
            break
    fig.supxlabel(' '.join((c.LABELS['perceptual noise name'], c.LABELS['perceptual noise symbol'], c.LABELS['location units'])))
    fig.supylabel(' '.join((c.LABELS['lag name'], c.LABELS['location units'])))

    return fig


def make_effective_timescale_fig(mle_data):
    def prepare_data(data):
        data = data.copy()
        data['effective_timescale'] = 1 / (data['K'] + data['b'])
        return data

    def plot(data):
        data = data.sort_index(level='speed')
        data_grouped = data.groupby('part_id')
        fig, axs = tile_subplots(len(data_grouped) + 1, figsize=(16, 16), sharex=True)
        for i, (part_id, part_data) in enumerate(data_grouped):
            ax = axs.flatten()[i]
            part_data = part_data.reset_index()
            ax.plot(part_data['speed'], part_data['effective_timescale'], marker='o', label=part_id)
            ax.set_title('')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(0, 1600)

        # plot median on last ax
        ax = axs.flatten()[-1]
        ax.plot(data.reset_index()['speed'].unique(), data['effective_timescale'].groupby('speed').median(), marker='o', color='k', label='median')

        fig.supxlabel(c.LABELS['mover speed name'] + ' ' + c.LABELS['speed units'])
        fig.supylabel("Effective timescale $(K+b)^{-1}$ [ms]")
        return fig

    mle_data = prepare_data(mle_data)
    fig = plot(mle_data)
    return fig


def main():
    parser = ArgumentParser()
    parser.add_argument('llk_path', type=str)
    parser.add_argument('bfc_prob_path', type=str)
    parser.add_argument('--path', default=None)
    args = parser.parse_args()

    set_defaults()
    set_local_defaults()
    bfc_vanilla = load_bfc_vanilla()
    bfc_et = load_bfc_et()
    mle_data = load_mle()
    llk_data = load_llks(args.llk_path)
    bfc_prob_data = load_bfc_llks(args.bfc_prob_path)

    figs_to_save = {
        'fidelity_all': make_fidelity_regressions_fig(mle_data),
        'lag_all_et': make_lag_vs_speed_fig(bfc_et)[0],
        'bfc_noise_all_et': make_bfc_noise_fig(bfc_et)[0],
        'psychometric_curves_vanilla': make_psychometric_curves_fig(bfc_vanilla),
        'psychometric_params_posterior_vanilla': make_psych_2d_posterior_fig(bfc_prob_data),
        'effective_timescale': make_effective_timescale_fig(mle_data),
        'individual_K': make_individual_K_fig(mle_data, llk_data),
    }

    figs = make_lag_vs_speed_fig(bfc_vanilla)
    for i, fig in enumerate(figs):
        figs_to_save[f'lag_vanilla_{i}'] = fig

    figs = make_bfc_noise_fig(bfc_vanilla)
    for i, fig in enumerate(figs):
        figs_to_save[f'bfc_noise_vanilla_{i}'] = fig

    figs = make_2d_likelihood_fig(llk_data)
    for i, fig in enumerate(figs):
        figs_to_save[f'2d_likelihood_{i}'] = fig

    figs = make_1d_likelihood_fig(llk_data)
    for i, fig in enumerate(figs):
        figs_to_save[f'1d_likelihood_{i}'] = fig

    # save figures
    if args.path is not None:
        for name, fig in figs_to_save.items():
            fig.savefig(os.path.join(args.path, f'{name}.pdf'))

    plt.show()


if __name__ == "__main__":
    main()
