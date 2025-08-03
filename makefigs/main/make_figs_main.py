"""
make_figs_main.py
Generate figures for the main text.

This script generates the figures for the main text of the paper. It uses the data from the Vanilla and ET experiments.
It relies on preprocessed data - see the usage section in the `README` file for details on how to preprocess the data.

Usage:
    python make_figs_main.py <data_path> [--output_path <output_path>]
Arguments:
    data_path: Path to the directory containing the data files.
    --output_path: Path to save the figures. Defaults to the current directory.

Example:
    python make_figs_main.py /path/to/data --output_path /path/to/save
"""

import makefigs.common as c
import matplotlib.pyplot as plt
import pandas as pd
from util.FLE_utils import regions_to_yerr
from ast import literal_eval
from ComDePy.viz import set_defaults
import sklearn as sk
import numpy as np
import seaborn as sns
from argparse import ArgumentParser
import os
import json
from scipy.odr import ODR, Model, RealData
from scipy.optimize import minimize
from scipy.stats import pearsonr, norm
from scipy.integrate import nquad
import statsmodels.api as sm
from analysis.DataAnalyzer import my_norm_CDF

mover_speed_label = ' '.join([c.LABELS['mover speed name'],
                              c.LABELS['mover speed symbol'],
                              c.LABELS['speed units']])


def set_local_defaults():
    plt.rcParams['font.size'] = 13


def load_bfc_data(path):
    cols = pd.read_csv(path, nrows=0).columns
    hdi_converter = {col: literal_eval for col in cols if col.endswith('_hdi')}
    dtype_convertes = {
        'part_id': str,
        'this_rep_n': int,
        'speed': float,
    }
    convertes = {**hdi_converter, **dtype_convertes}
    return pd.read_csv(path, index_col=['part_id', 'this_rep_n', 'speed'], converters=convertes)


def load_attention_check(path=None):
    return pd.read_csv(path, converters={'part_id': str}, index_col=['part_id'])


def load_vanilla_attention_check(path=None):
    if path is None:
        path = '../attn_check_vanilla.csv'
    return load_attention_check(path)


def load_et_attention_check(path=None):
    if path is None:
        path = '../attn_check_et.csv'
    return load_attention_check(path)


def load_bfc_et(path=None):
    if path is None:
        path = '../bfc_et.csv'
    return load_bfc_data(path)


def load_bfc_vanilla(path=None):
    if path is None:
        path = '../bfc_vanilla.csv'
    return load_bfc_data(path)


def load_mle(path=None):
    if path is None:
        path = '../et_mle.csv'
    return pd.read_csv(path, index_col=['part_id', 'speed'], dtype={'part_id': str, 'speed': float})


def exclude_bfc(bfc, attention_checks):
    """
    Exclude participants according to exclusion criteria.

    :param bfc: DataFrame with the BFC data.
    :return: DataFrame with the excluded participants removed.
    """

    def _parts_with_halfpoint_hdi_at_bounds(bfc):
        """
        Find participants with halfpoint HDI at bounds. They should be excluded because the inference is unreliable.
        """

        def _is_at_bound(hdi_values):
            return any(250.0 in t or -250.0 in t for t in hdi_values)

        invalid_part_ids = bfc.groupby('part_id')['halfpoint_hdi'].apply(
            lambda group: group.apply(_is_at_bound).any()
        )
        return invalid_part_ids[invalid_part_ids].index

    def _parts_with_negative_delay(bfc):
        """
        Find participants with negative delay. They should be excluded because their data is not modelled well by the
        extrapolation model.
        """

        def _has_negative_delay(data):
            model = estimate_delay(data)
            return model.coef_[0] < 0

        invalid_part_ids = bfc.groupby('part_id').apply(_has_negative_delay)
        return invalid_part_ids[invalid_part_ids].index

    def _parts_with_large_noise(bfc, threshold):
        """
        Find participants with large noise. They should be excluded because their data is not modelled well by the
        model.
        """
        has_large_noise = bfc['sd_est'] > threshold
        invalid_part_ids = bfc[has_large_noise].index.get_level_values('part_id').unique()
        return invalid_part_ids

    def _parts_with_many_failed_attention_checks(attention_checks, threshold):
        """
        Find participants with many failed attention checks. They should be excluded because they did not pay attention
        to the task.
        """
        # the result of the check is 0 if the participant failed the check, in column "success"
        failed_checks = attention_checks['success'] == 0
        n_failed_checks = failed_checks.groupby('part_id').sum()
        invalid_part_ids = n_failed_checks[n_failed_checks > threshold].index
        return invalid_part_ids

    exclusion_parameters = {
        'noise threshold (pixels)': c.BFC_NOISE_EXCLUSION_THRESHOLD,
        'noise uncertainty threshold (pixels)': c.BFC_NOISE_UNCERTAINTY_EXCLUSION_THRESHOLD,
        'attention check threshold': c.BFC_ATTENTION_CHECK_THRESHOLD
    }

    exclusion_reasons = {}
    halfpoint_at_bounds = _parts_with_halfpoint_hdi_at_bounds(get_latest_estimate(bfc))
    exclusion_reasons['halfpoint_at_bounds'] = halfpoint_at_bounds

    negative_delay = _parts_with_negative_delay(get_latest_estimate(bfc))
    exclusion_reasons['negative_delay'] = negative_delay

    large_noise = _parts_with_large_noise(get_latest_estimate(bfc),
                                          exclusion_parameters['noise threshold (pixels)'])
    exclusion_reasons['large_noise'] = large_noise

    # large_noise_uncertainty = _parts_with_large_noise_uncertainty(get_latest_estimate(bfc),
    #                                                               exclusion_parameters['noise uncertainty threshold (pixels)'])
    # exclusion_reasons['large_noise_uncertainty'] = large_noise_uncertainty

    many_failed_attention_checks = _parts_with_many_failed_attention_checks(attention_checks,
                                                                            exclusion_parameters['attention check threshold'])
    exclusion_reasons['many_failed_attention_checks'] = many_failed_attention_checks

    # Combine all
    invalid_part_ids = set()
    for reason, part_ids in exclusion_reasons.items():
        invalid_part_ids.update(part_ids)
    return bfc[~bfc.index.get_level_values('part_id').isin(invalid_part_ids)].copy(), exclusion_reasons, exclusion_parameters


def estimate_delay(bfc_data):
    """
    Estimate the delay for a single participant.

    :param bfc_data: DataFrame with the BFC data for a single participant.
    :return: The Bayesian Ridge model fitted for the data.
    """

    x = bfc_data.index.get_level_values('speed')
    y = bfc_data['halfpoint_est']
    w = calc_weight(std_from_hdi(bfc_data['halfpoint_hdi']))
    X = x.values.reshape(-1, 1)
    model = sk.linear_model.BayesianRidge()
    model.fit(X, y, w)
    return model


def get_latest_estimate(bfc_data):
    """
    Returns the latest estimate for each participant.

    :param bfc_data: DataFrame with the BFC data.
    :return: DataFrame with the latest estimate for each participant
    """
    max_n = bfc_data.index.get_level_values('this_rep_n').max()
    return bfc_data.loc[:, max_n, :].copy()


def std_from_hdi(hdi):
    """
    Calculate the effective error (standard deviation) given an HDI for a parameter.
    """
    hdi_size = hdi.apply(lambda x: (x[0][1] - x[0][0]) / 2)
    hdi_size[hdi_size == 0] = 2.5 / 2 / 2  # 2.5 is the resolution of the posterior domain.
    return hdi_size


def calc_weight(err):
    return 1 / err ** 2


def pearsonr_CI(x, y):
    """
    Calculate the Pearson correlation coefficient and the confidence interval for the correlation coefficient.

    :param x: Series with the first variable.
    :param y: Series with the second variable.
    :return: Tuple with the correlation coefficient and the confidence interval.
    """
    r, p = pearsonr(x, y)
    z = np.arctanh(r)
    se = 1 / np.sqrt(len(x) - 3)
    z_crit = norm.ppf(1 - 0.05 / 2)
    ci = np.tanh(z + np.array([-1, 1]) * z_crit * se).tolist()
    return r, ci, p


def approx_hessian(f, params, eps=1e-5):
    """
    Manually compute Hessian of f at 'params' via central differences.
    f: callable taking params (1D array) -> float
    params: 1D array of shape (n, )
    eps: small step for finite differences
    Returns: Hessian as an (n, n) numpy array
    """
    n = len(params)
    hessian = np.zeros((n, n), dtype=float)

    # Evaluate f at center
    f0 = f(params)

    for i in range(n):
        for j in range(i, n):
            # We need to shift param i and j by +/- eps
            # We'll do a 4-point evaluation for central difference:
            # f(x + e_i + e_j) + f(x - e_i - e_j) - f(x + e_i - e_j) - f(x - e_i + e_j)
            # Then divide by 4*eps_i*eps_j
            # Where e_i is eps in dimension i, 0 otherwise.

            if i == j:
                # We only shift param i
                # 2D version specialized for i=j:
                # f(x + e_i) + f(x - e_i) - 2 f0
                # divided by eps^2
                params_forward = params.copy()
                params_forward[i] += eps

                params_backward = params.copy()
                params_backward[i] -= eps

                f_forward = f(params_forward)
                f_backward = f(params_backward)

                second_deriv = (f_forward - 2*f0 + f_backward) / (eps**2)
                hessian[i, i] = second_deriv
            else:
                # i != j
                # We'll do the 2D central difference
                params_pp = params.copy()
                params_pp[i] += eps
                params_pp[j] += eps

                params_pm = params.copy()
                params_pm[i] += eps
                params_pm[j] -= eps

                params_mp = params.copy()
                params_mp[i] -= eps
                params_mp[j] += eps

                params_mm = params.copy()
                params_mm[i] -= eps
                params_mm[j] -= eps

                f_pp = f(params_pp)
                f_pm = f(params_pm)
                f_mp = f(params_mp)
                f_mm = f(params_mm)

                # central difference formula in 2D
                second_deriv = (f_pp + f_mm - f_pm - f_mp) / (4 * eps * eps)
                hessian[i, j] = second_deriv
                hessian[j, i] = second_deriv  # symmetric

    return hessian


def make_fig_1_psych_curve():
    x = np.linspace(-100, 200, 1000)
    mu, sigma = 100, 50
    y = my_norm_CDF(x, mu, sigma).squeeze()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: Empty panel with title
    axs[0].set_title('A', loc='left')
    axs[0].axis('off')

    # Panel B: Psychometric curve
    ax = axs[1]
    ax.set_title('B', loc='left', x=-0.2)
    ax.plot(x, y, 'k', label='Psychometric curve')
    ax.set_xlim(-100, 200)
    ax.set_ylim(-0.2, 1.1)

    # Add perceived lag annotation
    ax.annotate("", xy=(0, -0.15), xytext=(mu, -0.15),
                arrowprops=dict(arrowstyle='<->'))
    ax.text(mu / 2, -0.12, 'Perceived lag', ha='center', va='bottom')

    # Add perceptual noise annotation
    ax.annotate("", xy=(mu - sigma, 0.85), xytext=(mu + sigma, 0.85),
                arrowprops=dict(arrowstyle='<->'))
    ax.text(mu, 0.88, 'Perceptual noise', ha='center')

    # add dashed line from the x, y-axis to the curve at the halfpoint
    ax.plot([-100, mu], [0.5, 0.5], 'k--', lw=1)
    ax.plot([mu, mu], [-0.2, 0.5], 'k--', lw=1)

    # Configure ticks
    ax.set_xticks([0, mu])
    ax.set_xticklabels(['0', 'Halfpoint'])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # Labels
    ax.set_xlabel('Offset')
    ax.set_ylabel('Probability of "right" response')
    ax.grid(False)

    # Sample and plot responses
    np.random.seed(1)
    x_samples = np.random.uniform(-100, 200, 100)
    y_samples = my_norm_CDF(x_samples, mu, sigma).squeeze()
    responses = np.random.binomial(1, y_samples)
    ax.plot(x_samples, responses, 'o', label='Samples', alpha=0.5)

    return fig


def make_fig_2_lag_vs_speed(bfc_data, id):
    def _prepare_data_main(bfc_data, id):
        bfc_data = get_latest_estimate(bfc_data).loc[id, :, :]
        bfc_data = bfc_data.sort_index()
        model = estimate_delay(bfc_data)
        return bfc_data, model

    def _prepare_data_inset(bfc_data):
        bfc_data_filt = get_latest_estimate(bfc_data)
        bfc_data_filt['weight'] = calc_weight(std_from_hdi(bfc_data_filt['halfpoint_hdi']))
        # calculate the regression slope and uncertainty for each participant
        slopes = []
        stds = []
        pvalues = []
        for id, data in bfc_data_filt.groupby('part_id'):
            x = data.index.get_level_values('speed')
            y = data['halfpoint_est']
            w = data['weight']
            X = x.values.reshape(-1, 1)
            model = sk.linear_model.BayesianRidge()
            model.fit(X, y, w)
            slopes.append(model.coef_[0])
            stds.append(np.sqrt(model.sigma_[0, 0]))
            # calculate the p-value
            X = sm.add_constant(x)
            model_f = sm.WLS(y, X, weights=w).fit()
            pvalues.append(model_f.pvalues['x1'])
        pvalues = np.array(pvalues)
        max_pv = pvalues.max()
        df_res = pd.DataFrame({'slope': slopes, 'std': stds, 'pvalue': pvalues},
                              index=bfc_data.index.get_level_values('part_id').unique())
        return df_res, max_pv

    def _plot_main_panel(bfc_data, model):
        fig, ax = plt.subplots()

        x = bfc_data.index.get_level_values('speed')
        y = bfc_data['halfpoint_est']
        errs = regions_to_yerr(y.to_list(), bfc_data['halfpoint_hdi'].to_list())
        ax.errorbar(x, y, errs, fmt='o', label='Data')

        # Add fit
        slope = model.coef_[0] * 1000
        slope_std = np.sqrt(model.sigma_[0, 0]) * 1000
        x_2d = x.values.reshape(-1, 1)
        y_pred, y_std = model.predict(x_2d, return_std=True)
        ax.plot(x_2d, y_pred, 'k', lw=1,
                label='MLE fit',
                zorder=-1)
        ax.plot([x[0], x[-1]], [y_pred[0] + y_std[0], y_pred[-1] + y_std[-1]], 'k--', lw=1)
        ax.plot([x[0], x[-1]], [y_pred[0] - y_std[0], y_pred[-1] - y_std[-1]], 'k--', lw=1)

        ax.set_xlabel(mover_speed_label)
        ax.set_ylabel(c.LABELS['lag name'] + ' ' + c.LABELS['location units'])
        ax.set_title("")
        ax.set_xlim(0, 1600)
        ax.set_ylim(0, 180)
        ax.legend(loc='lower right')

        w = calc_weight(std_from_hdi(bfc_data['halfpoint_hdi']))
        fig_df = pd.DataFrame({'speed (pix/sec)': x, 'lag (pix)': y,
                               'lag error low (pix)': errs[0, :], 'lag error high (pix)': errs[1, :], 'weight': w})
        return fig, slope, slope_std, fig_df

    def _plot_inset(fig, bfc_data):
        left, bottom, width, height = [0.185, 0.56, 0.38, 0.38]
        ax = fig.add_axes([left, bottom, width, height])
        # Plot the distribution of slopes
        sns.histplot(bfc_data['slope'] * 1000, ax=ax, kde=False)
        ax.set_ylim(0, 25)
        ax.set_xlabel('')
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.set_ylabel('Count', fontsize='small')
        ax.set_title('All participants\' delays', fontsize='small', y=1, pad=-13)
        ax.set_xticks([0, 50, 100], ["0ms", "50ms", "100ms"], minor=False, rotation=35, fontsize='small')
        ax.set_xticks([], minor=True)
        ax.set_yticks([0, 5, 10, 15, 20], ["0", "5", "10", "15", "20"], minor=False, fontsize='small')
        ax.set_yticks([], minor=True)
        ax.grid(visible=None)
        fig_df = pd.DataFrame({'slope (ms)': bfc_data['slope'] * 1000,
                               'std': bfc_data['std'], 'pvalue': bfc_data['pvalue']})
        return fig, fig_df

    bfc_data_one_part, fit = _prepare_data_main(bfc_data, id)
    fig, slope, slope_std, main_df = _plot_main_panel(bfc_data_one_part, fit)
    inset_data, max_pv = _prepare_data_inset(bfc_data)
    fig, inset_df = _plot_inset(fig, inset_data)
    fig_data = {"delay (ms)": slope, "delay std (ms)": slope_std, 'max p-value': max_pv}
    return fig, fig_data, [main_df, inset_df]


def make_fig_3_bfc_noise(bfc_data):
    def _prepare_data(bfc_data):
        bfc_data = get_latest_estimate(bfc_data)
        bfc_data['weight'] = calc_weight(std_from_hdi(bfc_data['sd_hdi']))
        bfc_data = bfc_data.sort_index()
        x = np.log(bfc_data.index.get_level_values('speed'))
        y = np.log(bfc_data['sd_est'])
        X = x.values.reshape(-1, 1)
        w = bfc_data['weight'] * bfc_data['sd_est'] ** 2  # convert to log space
        model = sk.linear_model.BayesianRidge()
        model.fit(X, y, w)
        # calculate the p-value
        X = sm.add_constant(x)
        model_f = sm.WLS(y, X, weights=w).fit()
        pvalue = model_f.pvalues['x1']
        fit_data = {"slope": model.coef_[0], "slope_std": np.sqrt(model.sigma_[0, 0]), "pvalue": pvalue}
        return bfc_data, model, fit_data

    def _plot(bfc_data, model, ax):
        # Plot the mean
        x = bfc_data.index.get_level_values('speed').unique()
        y = bfc_data.groupby('speed')['sd_est'].median()
        ax.plot(x, y, 'o', label='Median', markersize=10)

        # Plot all data
        x = bfc_data.index.get_level_values('speed')
        y = bfc_data['sd_est']
        ax.plot(x, y, 'D', label='Data', c='C2', markersize=5, alpha=0.3, zorder=-1)

        # Add fit
        x_2d = np.log(x.unique().values.reshape(-1, 1))
        y_pred, y_std = model.predict(x_2d, return_std=True)
        ax.plot(x.unique(), np.exp(y_pred), 'k', lw=1,
                label=fr'MLE fit',
                zorder=-1)

        # Add uncertainty lines
        ax.plot(x.unique(), np.exp(y_pred + y_std), 'k--', lw=1)
        ax.plot(x.unique(), np.exp(y_pred - y_std), 'k--', lw=1)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(mover_speed_label)
        ax.set_ylabel(" ".join((c.LABELS['perceptual noise name'],
                                c.LABELS['perceptual noise symbol'],
                                c.LABELS['location units'])))
        ax.set_title("A", loc='left', x=-0.165)
        ax.legend(fontsize='small')

        w = calc_weight(std_from_hdi(bfc_data['sd_hdi']))
        w_log = bfc_data['weight'] * bfc_data['sd_est'] ** 2
        fig_df = pd.DataFrame({'speed (pix/sec)': x, 'noise (pix)': y, 'weight': w, 'weight in log space': w_log})
        return ax, fig_df

    def _plot_sketch(ax, sketch_path):
        ax.axis('off')
        ax.set_title("B", loc='left')

    bfc_data, model, fit_data = _prepare_data(bfc_data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _, fig_df = _plot(bfc_data, model, axes[0])
    _plot_sketch(axes[1], 'sqrt law sketch.png')
    return fig, fit_data, fig_df


def make_fig_4_K(mle_data):
    def _prepare_data(data):
        # ----------------
        # Define parameter ranges
        # ----------------
        slope_lower, slope_upper = 0, 1e-5
        intercept_lower, intercept_upper = 0.001, 0.01
        sigma_lower, sigma_upper = 1e-4, 0.1
        region1 = [(intercept_lower, intercept_upper), (sigma_lower, sigma_upper)]
        region2 = [(slope_lower, slope_upper), (intercept_lower, intercept_upper), (sigma_lower, sigma_upper)]

        # ----------------
        # Define the data
        # ----------------
        x = data.index.get_level_values('speed')
        y = data['K']
        n_obs = len(y)

        # ----------------
        # Define negative log-likelihood functions
        # ----------------
        def nll_model1(params):
            b, sigma = params
            residuals = y - b
            return -np.sum(norm.logpdf(residuals, loc=0, scale=sigma))

        def nll_model2(params):
            a, b, sigma = params
            residuals = y - a * x - b
            return -np.sum(norm.logpdf(residuals, loc=0, scale=sigma))

        # ----------------
        # Fit the models
        # ----------------
        intercept_0 = (intercept_upper - intercept_lower) / 2
        sigma_0 = (sigma_upper - sigma_lower) / 2
        slope_0 = (slope_upper - slope_lower) / 2
        res1 = minimize(nll_model1,
                        x0=[intercept_0, sigma_0],
                        bounds=region1)
        res2 = minimize(nll_model2,
                        x0=[slope_0, intercept_0, sigma_0],
                        bounds=region2)
        if not (res1.success and res2.success):
            raise RuntimeError("MLE optimization failed - "
                               f"Model 1: {res1.message}, Model 2: {res2.message}")

        # ----------------
        # Calculate AIC and BIC
        # ----------------
        k1, k2 = 2, 3
        ll1 = -nll_model1(res1.x)
        ll2 = -nll_model2(res2.x)
        aic1 = 2 * k1 - 2 * ll1
        aic2 = 2 * k2 - 2 * ll2
        bic1 = k1 * np.log(n_obs) - 2 * ll1
        bic2 = k2 * np.log(n_obs) - 2 * ll2

        # ----------------
        # Estimate uncertainties in parameters
        # ----------------
        H1 = approx_hessian(nll_model1, res1.x)
        H2 = approx_hessian(nll_model2, res2.x)
        # calculate the covariance matrix
        cov1 = np.linalg.inv(H1)
        cov2 = np.linalg.inv(H2)
        # calculate the standard deviations
        std1 = np.sqrt(np.diag(cov1))
        std2 = np.sqrt(np.diag(cov2))
        mle1 = {'b': res1.x[0], 'b_std': std1[0], 'sigma': res1.x[1], 'sigma_std': std1[1]}
        mle2 = {'a': res2.x[0], 'a_std': std2[0], 'b': res2.x[1], 'b_std': std2[1], 'sigma': res2.x[2], 'sigma_std': std2[2]}

        # ----------------
        # Define priors (Jeffreys 1/sigma x Uniform)
        # ----------------
        def prior1(b, sigma):
            if intercept_lower <= b <= intercept_upper and sigma_lower <= sigma <= sigma_upper:
                return 1.0 / sigma
            else:
                return 0.0
        n1 = nquad(prior1, region1)[0]

        def prior2(a, b, sigma):
            if (slope_lower <= a <= slope_upper and intercept_lower <= b <= intercept_upper
                    and sigma_lower <= sigma <= sigma_upper):
                return 1.0 / sigma
            else:
                return 0.0
        n2 = nquad(prior2, region2)[0]

        # ----------------
        # Evidence integrands
        # ----------------
        def integrand1(b, sigma):
            ll = np.exp(-nll_model1([b, sigma]))
            return prior1(b, sigma) * ll / n1

        def integrand2(a, b, sigma):
            ll = np.exp(-nll_model2([a, b, sigma]))
            return prior2(a, b, sigma) * ll / n2
        evidence1 = nquad(integrand1, region1)[0]
        evidence2 = nquad(integrand2, region2)[0]
        bf = evidence1 / evidence2 if evidence2 != 0 else np.inf
        return {
            "Model 1 MLE": mle1,
            "Model 2 MLE": mle2,
            "Bayes factor for model 1": bf,
            "Model 1 AIC": aic1,
            "Model 2 AIC": aic2,
            "Model 1 BIC": bic1,
            "Model 2 BIC": bic2,
        }

    def _prepare_data_inset(data):
        data = data.copy()
        data['weight'] = 1  # Fixme: add weight calculation
        slopes = []
        stds = []
        for id, part_data in data.groupby('part_id'):
            x = part_data.index.get_level_values('speed')
            y = part_data['K']
            w = part_data['weight']
            X = x.values.reshape(-1, 1)
            model = sk.linear_model.BayesianRidge()
            model.fit(X, y, w)
            slopes.append(model.coef_[0])
            stds.append(np.sqrt(model.sigma_[0, 0]))
        return pd.DataFrame({'slope': slopes, 'std': stds}, index=data.index.get_level_values('part_id').unique())

    def _plot_main(data):
        fig, ax = plt.subplots()
        for ii, (id, part_data) in enumerate(data.groupby('part_id')):
            part_data = part_data.sort_index()
            x = part_data.index.get_level_values('speed')
            y = part_data['K']
            ax.plot(x, y, '-o', c='C0',
                    alpha=0.25,
                    )
        # add median
        part_mle = mle_data.groupby('speed').median()
        ax.plot(part_mle.index, part_mle['K'], 'k--o', label='Median')

        ax.set_yscale('log')
        ax.set_xlabel(mover_speed_label)
        ax.set_ylabel(' '.join([c.LABELS['K name'],
                                c.LABELS['K symbol'],
                                c.LABELS['K units']]))
        ax.set_title('')

        fig_df = pd.DataFrame({'participant id': data.index.get_level_values('part_id'),
                               'speed (pix/sec)': data.index.get_level_values('speed'),
                               'K (1/sec)': data['K']})
        return fig, fig_df

    def _plot_inset(fig, data):
        left, bottom, width, height = [0.65, 0.25, 0.25, 0.2]
        ax = fig.add_axes([left, bottom, width, height])
        # Plot the distribution of slopes
        sns.histplot(data['slope'], ax=ax, kde=False)

        # labels
        ax.set_title("", fontsize='small')
        ax.set_xlabel('Slope [1/Pixel]', fontsize='small')
        ax.set_ylabel('Count', fontsize='small')

        # ticks
        ax.ticklabel_format(axis='x', style='plain')
        ax.set_xticks([0, 1e-5], ["0", r"$10^{-5}$"], fontsize='small')
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)

        # grid
        ax.grid(visible=None)

        fig_df = pd.DataFrame({'slope (1/pix)': data['slope']})
        return fig, fig_df

    fig_stats = _prepare_data(mle_data)
    inset_data = _prepare_data_inset(mle_data)
    fig, fig_df = _plot_main(mle_data)
    fig, inset_df = _plot_inset(fig, inset_data)
    return fig, fig_stats, [fig_df, inset_df]


def make_fig_5_fidelity(mle_data):
    def _prepare_data(data):
        data = data.copy()
        data['fidelity'] = data['K'] / (data['K'] + data['b'])
        return data

    def _prepare_data_inset(data):
        data = data.copy()
        data['weight'] = 1  # Fixme: add weight calculation
        slopes = []
        stds = []
        for id, part_data in data.groupby('part_id'):
            x = part_data.index.get_level_values('speed')
            y = part_data['fidelity']
            w = part_data['weight']
            X = x.values.reshape(-1, 1)
            model = sk.linear_model.BayesianRidge()
            model.fit(X, y, w)
            slopes.append(model.coef_[0])
            stds.append(np.sqrt(model.sigma_[0, 0]))
        return pd.DataFrame({'slope': slopes, 'std': stds}, index=data.index.get_level_values('part_id').unique())

    def _plot(data):
        fig, ax = plt.subplots()
        for ii, (id, part_data) in enumerate(data.groupby('part_id')):
            part_data = part_data.sort_index()
            x = part_data.index.get_level_values('speed')
            y = part_data['fidelity']
            ax.plot(x, y, '-o', c='C0',
                    alpha=0.25,
                    )
        # add median
        part_mle = mle_data.groupby('speed').median()
        ax.plot(part_mle.index, part_mle['fidelity'], 'k--o', label='Median')

        ax.axhline(1, color='k', linestyle='--', label='Perfect fidelity', alpha=0.25)
        ax.set_xlabel(mover_speed_label)
        ax.set_ylabel(' '.join([c.LABELS['fidelity name'],
                                c.LABELS['fidelity symbol']]))
        ax.set_title('')
        ax.set_ylim(0.4, 1.02)
        ax.grid(visible=None)

        fig_df = pd.DataFrame({'participant id': data.index.get_level_values('part_id'),
                               'speed (pix/sec)': data.index.get_level_values('speed'),
                               'fidelity': data['fidelity']})
        return fig, fig_df

    def _plot_inset(fig, data):
        left, bottom, width, height = [0.25, 0.22, 0.25, 0.2]
        ax = fig.add_axes([left, bottom, width, height])
        # Plot the distribution of slopes
        sns.histplot(data['slope'], ax=ax, kde=False)

        # labels
        ax.set_title("", fontsize='small')
        ax.set_xlabel('Slope [Seconds/Pixel]', fontsize='small')
        ax.set_ylabel('Count', fontsize='small')

        # ticks
        ax.ticklabel_format(axis='x', style='plain')
        ax.set_xticks([-5e-4, -2.5e-4, 0, 1.3e-4],
                      ["-5", "-2.5", "0", r"$\times 10^{-4}$"], fontsize='smaller')
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)

        # grid
        ax.grid(visible=None)

        fig_df = pd.DataFrame({'slope (sec/pix)': data['slope']})
        return fig, fig_df

    mle_data = _prepare_data(mle_data)
    inset_data = _prepare_data_inset(mle_data)
    fig, fig_df = _plot(mle_data)
    fig, inset_df = _plot_inset(fig, inset_data)
    return fig, [fig_df, inset_df]


def make_fig_6_sig_sig(bfc, mle):
    def _prepare_data(bfc, mle):
        bfc = get_latest_estimate(bfc).sort_index()
        # filter out participant speeds with large noise uncertainty
        bfc = bfc[bfc['sd_hdi'].apply(lambda x: x[0][1] - x[0][0] < c.BFC_NOISE_UNCERTAINTY_EXCLUSION_THRESHOLD)]
        # calculate the delay
        delays = {}
        delays_stds = {}
        for id, part_data in bfc.groupby('part_id'):
            model = estimate_delay(part_data)
            delays[id] = model.coef_[0] * 1000
            delays_stds[id] = np.sqrt(model.sigma_[0, 0]) * 1000
        mle = mle.copy()
        mle['tau_effective'] = 1 / (mle['K'] + mle['b'])
        mle['sig_r_effective'] = (mle['sig_r'] *
                                  np.sqrt(mle['tau_effective']) *
                                  mle.index.get_level_values('part_id').map(delays))
        mle['sig_r_effective_std'] = (mle['sig_r'] *
                                      np.sqrt(mle['tau_effective']) *
                                      mle.index.get_level_values('part_id').map(delays_stds))  # FIXME: add uncertainties of sig_r and tau_eff for full error propagation

        # Fit a linear model
        def model_func(p, x):
            return p[0] * x + p[1]

        model = Model(model_func)
        x = bfc['sd_est']
        y = mle['sig_r_effective']
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        x_err = std_from_hdi(bfc.loc[common_idx]['sd_hdi'])
        y_err = mle.loc[common_idx]['sig_r_effective_std']
        data = RealData(x, y, sx=x_err, sy=y_err)
        odr = ODR(data, model, beta0=[1, 0])
        fit = odr.run()

        rho, ci, p_value = pearsonr_CI(x, y)
        return bfc, mle, fit, {"rho": rho, "rho_CI": ci, "p_value": p_value}

    def _plot(bfc, mle, fit):
        fig, ax = plt.subplots()
        x = bfc['sd_est']
        y = mle['sig_r_effective']
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        x_err = regions_to_yerr(x.to_list(), bfc.loc[common_idx]['sd_hdi'].to_list())
        y_err = mle.loc[common_idx]['sig_r_effective_std']
        ax.plot(x, y, 'o', c='C0', label='Data')

        # Add fit
        sns.regplot(x=x, y=y, ax=ax, scatter=False, label="Regression line",
                    line_kws={'color': 'k', 'linewidth': 1})
        ax.legend()

        ax.set_xlim(0, 55)
        ax.set_ylim(0, 55)

        ax.set_xlabel(' '.join([c.LABELS['perceptual noise name'],
                                c.LABELS['perceptual noise symbol'],
                                c.LABELS['location units']]))
        ax.set_ylabel(' '.join(['Effective', c.LABELS['et noise name'],
                                c.LABELS['et noise symbol'],
                                c.LABELS['location units']]))

        fig_df = pd.DataFrame({'perceptual noise (pix)': x, 'effective et noise (pix)': y,
                               'perceptual noise error low (pix)': x_err[0, :],
                               'perceptual noise error high (pix)': x_err[1, :],
                               'effective et noise error low (pix)': y_err,
                               'effective et noise error high (pix)': y_err})
        return fig, fig_df

    bfc, mle, fit, pearson = _prepare_data(bfc, mle)
    fit_data = {"slope": fit.beta[0], "slope_std": fit.sd_beta[0], "intercept": fit.beta[1],
                "intercept_std": fit.sd_beta[1],
                "rho": pearson["rho"], "rho_CI": pearson["rho_CI"], "p_value": pearson["p_value"]}
    fig, fig_data = _plot(bfc, mle, fit)
    return fig, fit_data, fig_data


def run_main():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    args = parser.parse_args()

    set_defaults()
    set_local_defaults()

    # load eye tracking binary forced-choice data
    print("loading ET data")
    bfc_et = load_bfc_et()
    bfc_et_checks = load_et_attention_check()
    bfc_et, exc_reasons_et, exc_params = exclude_bfc(bfc_et, bfc_et_checks)
    if args.path is not None:
        with open(os.path.join(args.path, 'exclusions_et.json'), 'w') as f:
            exc_reasons_et = {k: list(v) for k, v in exc_reasons_et.items()}
            json.dump(exc_reasons_et, f, indent=4)
        with open(os.path.join(args.path, 'exclusion_params_et.json'), 'w') as f:
            json.dump(exc_params, f, indent=4)

    # load vanilla binary forced-choice data
    print("loading vanilla data")
    bfc_vanilla = load_bfc_vanilla()
    bfc_vanilla_checks = load_vanilla_attention_check()
    bfc_vanilla, exc_reasons_vanilla, _ = exclude_bfc(bfc_vanilla, bfc_vanilla_checks)
    if args.path is not None:
        with open(os.path.join(args.path, 'exclusions_vanilla.json'), 'w') as f:
            exc_reasons_vanilla = {k: list(v) for k, v in exc_reasons_vanilla.items()}
            json.dump(exc_reasons_vanilla, f, indent=4)

    # load eye tracking MLE
    print("loading ET MLE data")
    mle_et = load_mle()
    # exclude participant ids that are not in the bfc data
    mle_et = mle_et[mle_et.index.get_level_values('part_id').isin(bfc_et.index.get_level_values('part_id'))]

    # Generate figures
    figs_to_save = {}
    fig_df_to_save = {}

    print("Generating figures")

    print("Figure 1...")
    fig = make_fig_1_psych_curve()
    figs_to_save['fig_1'] = [fig]

    print("Figure 2...")
    lag_vs_speed_exemplar_id = "21793343"
    fig, fig_data, fig_df = make_fig_2_lag_vs_speed(bfc_vanilla, lag_vs_speed_exemplar_id)
    figs_to_save['lag_vs_speed'] = [fig, fig_data]
    fig_df_to_save['fig 2'] = fig_df[0]
    fig_df_to_save['fig 2 inset'] = fig_df[1]

    print("Figure 3...")
    fig, fit_data, fig_df = make_fig_3_bfc_noise(bfc_vanilla)
    figs_to_save['bfc_noise'] = [fig, fit_data]
    fig_df_to_save['fig 3'] = fig_df

    print("Figure 4...")
    fig, fig_stats, fig_df = make_fig_4_K(mle_et)
    figs_to_save['K'] = [fig, fig_stats]
    fig_df_to_save['fig 4'] = fig_df[0]
    fig_df_to_save['fig 4 inset'] = fig_df[1]

    print("Figure 5...")
    fig, fig_df = make_fig_5_fidelity(mle_et)
    figs_to_save['fidelity'] = [fig]
    fig_df_to_save['fig 5'] = fig_df[0]
    fig_df_to_save['fig 5 inset'] = fig_df[1]

    print("Figure 6...")
    fig, fit_data, fig_df = make_fig_6_sig_sig(bfc_et, mle_et)
    figs_to_save['sig-sig_r'] = [fig, fit_data]
    fig_df_to_save['fig 6'] = fig_df
    plt.show()

    # save figures and data
    print("Saving figures and data")
    if args.path is not None:
        for name, fig_data in figs_to_save.items():
            fig_data[0].savefig(os.path.join(args.path, f'{name}.pdf'))
            if len(fig_data) > 1:
                with open(os.path.join(args.path, f'{name}.json'), 'w') as f:
                    json.dump(fig_data[1], f, indent=4)
        if fig_df_to_save:
            writer = pd.ExcelWriter(os.path.join(args.path, 'data.xlsx'))
            for name, data in fig_df_to_save.items():
                data.to_excel(writer, sheet_name=name, index=False, engine='xlsxwriter')
            writer.close()

    print("Done!")


if __name__ == '__main__':
    run_main()
