#  This script tests what happens when the estimation models doesn't fit the data generating model in various ways.

# TODO replace uncertainty with HDI (ArviZ)
import numpy as np
from questplus.psychometric_function import weibull, norm_cdf
from questplus.qp import QuestPlusWeibull
from util.qp_extend import QuestPlusNorm
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from util import FLE_utils
from tqdm import trange
from ComDePy.viz import set_defaults

set_defaults()
mpl.rcParams['axes.grid'] = True
warnings.filterwarnings('ignore', category=RuntimeWarning)  # ignore runtime warnings coming up due to log(0) in the
PM = u'\u00b1'  # plus-minus sign
SEED = 123
RNG = np.random.default_rng(SEED)
FUNC = 'norm_cdf'
if FUNC == 'weibull':
    STIM_SCALE = 'log10'
    LOC_PARAM_NAME = 'threshold'
    SCALE_PARAM_NAME = 'slope'
elif FUNC == 'norm_cdf':
    STIM_SCALE = 'linear'
    LOC_PARAM_NAME = 'mean'
    SCALE_PARAM_NAME = 'sd'
HIGH_RESPONSE = 'high'
LOW_RESPONSE = 'low'


class IdealParticipant:
    def __init__(self, halfpoint, trans_region, asymptote=0.0, lapse_rate=0.0, func='weibull', stim_scale=None):
        self.halfpoint = FLE_utils.halfpoint_to_threshold(halfpoint, trans_region, stim_scale)
        self.trans_region = trans_region
        self.asymptote = asymptote
        self.lapse_rate = lapse_rate
        function = None
        if func == 'weibull':
            if stim_scale is None:
                stim_scale = 'log10'
            slope_ = 1 / self.trans_region
            thresh = FLE_utils.halfpoint_to_threshold(halfpoint, slope_, stim_scale)
            function = lambda x: weibull(intensity=x, threshold=thresh, slope=slope_,
                                         lower_asymptote=self.asymptote, lapse_rate=self.lapse_rate,
                                         scale=stim_scale)
        elif func == 'norm_cdf':
            if stim_scale is None:
                stim_scale = 'linear'
            function = lambda x: norm_cdf(intensity=x, mean=self.halfpoint, sd=self.trans_region,
                                          lower_asymptote=self.asymptote, lapse_rate=self.lapse_rate,
                                          scale=stim_scale)
        self.stim_scale = stim_scale
        self.psych_function = function

    def get_response_prob(self, intensity):
        return self.psych_function(intensity)

    def get_response(self, intensity):
        prob_corr = self.get_response_prob(intensity)
        dice_roll = RNG.random()
        if dice_roll <= prob_corr:
            return HIGH_RESPONSE
        else:
            return LOW_RESPONSE


# set up domains
density = 101
int_max = 100
int_min = -int_max
int_domain = np.linspace(int_min, int_max, density)

locs = np.linspace(int_min, int_max, density)
scales = np.logspace(0, 2.5, density)

# set up idealized participant
halfpoint = 0
trans_region = 20
print(f'Real halfpoint: {halfpoint}, transition region: {trans_region}')
p_bitflip = 0.0
lower_asy_participant = 0
lapse_rate_participant = 0
participant = IdealParticipant(halfpoint, trans_region, asymptote=lower_asy_participant,
                               lapse_rate=lapse_rate_participant,
                               stim_scale=STIM_SCALE, func=FUNC)

# set up Quest+ object
p_bitflip_sampler = 0
lower_asy_sampler = 0
lapse_rate_sampler = 0
shared_qp_params = {
    'intensities': int_domain,
    'lower_asymptotes': None,
    'lapse_rates': None,
    'responses': [HIGH_RESPONSE, LOW_RESPONSE],
    'stim_scale': STIM_SCALE,
    'stim_selection_method': 'min_entropy',
    'param_estimation_method': 'mean',
}
if FUNC == 'weibull':
    shared_qp_params['thresholds'] = locs
    shared_qp_params['slopes'] = 1 / scales
elif FUNC == 'norm_cdf':
    shared_qp_params['means'] = locs
    shared_qp_params['sds'] = scales
sampler_params = shared_qp_params.copy()
sampler_params['lower_asymptotes'] = [lower_asy_sampler]
sampler_params['lapse_rates'] = [lapse_rate_sampler]
if FUNC == 'weibull':
    sampler = QuestPlusWeibull(**sampler_params)
elif FUNC == 'norm_cdf':
    sampler = QuestPlusNorm(**sampler_params)

# generate data
n_trials = 100
intensities = []
responses = []
halfpoint_estimates = []
scale_estimates = []
print('Running trials...')
for trial in trange(n_trials):
    intensity = sampler.next_intensity
    response = participant.get_response(intensity)
    sampler.update(intensity=intensity, response=response)
    intensities.append(intensity)
    responses.append(response)
    if FUNC == 'weibull':
        halfpoint_estimate = FLE_utils.threshold_to_halfpoint(sampler.param_estimate[LOC_PARAM_NAME],
                                                              sampler.param_estimate[SCALE_PARAM_NAME], STIM_SCALE)
    elif FUNC == 'norm_cdf':
        halfpoint_estimate = sampler.param_estimate[LOC_PARAM_NAME]
    halfpoint_estimates.append(halfpoint_estimate)
    scale_estimates.append(sampler.param_estimate[SCALE_PARAM_NAME])
halfpoint_estimates = np.asarray(halfpoint_estimates)
scale_estimates = np.asarray(scale_estimates)
if FUNC == 'weibull':
    transition_widths = 1 / scale_estimates
elif FUNC == 'norm_cdf':
    transition_widths = scale_estimates

if FUNC == 'weibull':
    (tv, sv) = np.meshgrid(sampler.thresholds, sampler.slopes)
    halfpoint_mesh = FLE_utils.threshold_to_halfpoint(tv, sv, STIM_SCALE)
    red_posterior = sampler.posterior.squeeze()
    halfpoint_estimate_final_unc = FLE_utils.calc_sd(red_posterior, halfpoint_mesh)
    trans_width_estimates_final_unc = FLE_utils.calc_sd(sampler.marginal_posterior[SCALE_PARAM_NAME], 1 / sampler.slopes)
elif FUNC == 'norm_cdf':
    halfpoint_estimate_final_unc = FLE_utils.calc_sd(sampler.marginal_posterior[LOC_PARAM_NAME], sampler.means)
    trans_width_estimates_final_unc = FLE_utils.calc_sd(sampler.marginal_posterior[SCALE_PARAM_NAME], sampler.sds)
print('====Sampler results====')
print(f'Halfpoint estimate: {int(halfpoint_estimates[-1])}{PM}{int(halfpoint_estimate_final_unc)}, '
      f'Transition width estimate: {int(transition_widths[-1])}{PM}{int(trans_width_estimates_final_unc)}')
print(f'"right" response fraction: {responses.count(HIGH_RESPONSE) / len(responses):.2g}')

# re-estimate using post-hoc estimator
p_bitflip_estimator = 0
estimator_params = shared_qp_params.copy()
estimator_params['lower_asymptotes'] = [p_bitflip_estimator]
estimator_params['lapse_rates'] = [p_bitflip_estimator]
if FUNC == 'weibull':
    estimator = QuestPlusWeibull(**estimator_params)
elif FUNC == 'norm_cdf':
    estimator = QuestPlusNorm(**estimator_params)
for intensity, response in zip(intensities, responses):
    estimator.update(intensity=intensity, response=response)
if FUNC == 'weibull':
    halfpoint_re_estimate = FLE_utils.threshold_to_halfpoint(estimator.param_estimate[LOC_PARAM_NAME],
                                                             estimator.param_estimate[SCALE_PARAM_NAME], STIM_SCALE)
    halfpoint_re_estimate_unc = FLE_utils.calc_sd(red_posterior, halfpoint_mesh)
    slope_re_estimate = estimator.param_estimate[SCALE_PARAM_NAME]
    slope_re_estimate_unc = FLE_utils.calc_sd(estimator.marginal_posterior[SCALE_PARAM_NAME], estimator.slopes)
    transition_width_re_estimate = 1 / slope_re_estimate
    transition_width_re_estimate_unc = FLE_utils.calc_sd(estimator.marginal_posterior[SCALE_PARAM_NAME],
                                                         1 / estimator.slopes)
elif FUNC == 'norm_cdf':
    halfpoint_re_estimate = estimator.param_estimate[LOC_PARAM_NAME]
    halfpoint_re_estimate_unc = FLE_utils.calc_sd(estimator.marginal_posterior[LOC_PARAM_NAME], estimator.means)
    transition_width_re_estimate = estimator.param_estimate[SCALE_PARAM_NAME]
    transition_width_re_estimate_unc = FLE_utils.calc_sd(estimator.marginal_posterior[SCALE_PARAM_NAME], estimator.sds)
print('====Re-estimator results====')
print(f'Halfpoint re-estimate: {int(halfpoint_re_estimate)}{PM}{int(halfpoint_re_estimate_unc)}, '
      f'Transition width re-estimate: {int(transition_width_re_estimate)}{PM}{int(transition_width_re_estimate_unc)}')

# plot data
plt.figure()
low_color = 'C0'
high_color = 'C1'
colors = [low_color if response == LOW_RESPONSE else high_color for response in responses]
plt.scatter(range(n_trials), intensities, c=colors, s=60)
plt.plot(range(n_trials), intensities, c='k', alpha=0.5)

# plot psychometric function parameters, real and estimated
plt.axhline(halfpoint, c='C2', ls=':', label='Halfpoint')
upper = halfpoint + 1 / 2 * trans_region
lower = halfpoint - 1 / 2 * trans_region
plt.fill_between(range(n_trials, n_trials + 2), lower, upper, color='C2', alpha=0.8, label='Transition width')

upper_est = halfpoint_estimates + 1 / 2 * transition_widths
lower_est = halfpoint_estimates - 1 / 2 * transition_widths
plt.plot(range(n_trials), halfpoint_estimates, c='k', ls='--', label='Halfpoint estimate')
plt.fill_between(range(n_trials), lower_est, upper_est, color='k', alpha=0.2, label='Transition width estimate')

plt.xlabel('Trial')
plt.ylabel('Intensity')
l1 = Line2D([0], [0], color=low_color, marker='o', linestyle='None')
l2 = Line2D([0], [0], color=high_color, marker='o', linestyle='None')
l3 = Line2D([0], [0], color='k', ls='--')
l4 = Line2D([0], [0], color='C2', ls=':')
plt.legend((l1, l2, l3, l4), ('Left', 'Right', 'Estimate', 'Real'), loc='lower right')
plt.xlim(-1, n_trials + 1)

plt.title(f'Real $P_{{bitflip}}$ = {p_bitflip}, model $P_{{bitflip}}$ = {p_bitflip_sampler}')

# plot with psych function
plt.figure()
plt.plot(int_domain, participant.psych_function(int_domain).squeeze(), c='k', label='Real psychometric function')
plt.scatter(intensities, [1 if r == HIGH_RESPONSE else 0 for r in responses], c=colors, s=60)
plt.xlabel('Intensity')
plt.ylabel('Probability of response = \'right\'')
plt.legend()

plt.show()
