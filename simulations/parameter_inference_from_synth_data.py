# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test inferrers
# In this notebook I try to see how much data is needed to accurately extract the parameter(s) in different conditions. I will define common parameter ranges, number of samples, number of trials. Extraction will be tested by maximizing the likelihood of the data given the model.

# %%
# Imports
import numpy as np
import xarray as xr
from tqdm.notebook import tqdm
from collections import OrderedDict
from scipy.optimize import minimize
from analysis.bayesian_inference_et import PosControlKernel, PosControlMeanProcess, VelControlKernel, VelControlMeanProcess
from analysis.gaussian_process import llk

from ComDePy.viz import set_defaults
set_defaults()

# %%
DEF_PARAMS = {
    'V': 0.5,
    'K': 0.1,
    'sig_r': 0.05,
    'sig_y': 0.05,
    'gamma': 0,
    'b': 0.1
}

OPT_BOUNDS = {
    'K': (1e-10, None),
    'sig_r': (1e-10, None),
    'sig_y': (1e-10, None),
    'gamma': (0, 0),
    'b': (1e-10, 0.5)
}

T_MAX = 100
N_SAMPLES_RANGE = [10, 50, 100]
N_TRIALS_RANGE = [1, 10, 100, 1000]


def nll_pos(K, sig_r, sig_y, gamma, t, y):
    mean_proc = PosControlMeanProcess(K)
    kernel = PosControlKernel(K, sig_r, sig_y, gamma)
    mu = mean_proc(t, v=DEF_PARAMS['V'])
    return -llk(t, y, mu, kernel)

def nll_sum_pos(K, sig_r, sig_y, gamma, t, Y):
    sum = 0
    for y in Y:
        sum += nll_pos(K, sig_r, sig_y, gamma, t, y)
    return sum

def nll_vel(K, b, sig_r, sig_y, t, y):
    mean_proc = VelControlMeanProcess(K, b)
    kernel = VelControlKernel(K, b, sig_r, sig_y)
    mu = mean_proc(t, v=DEF_PARAMS['V'])
    return -llk(t, y, mu, kernel)

def nll_sum_vel(K, b, sig_r, sig_y, t, Y):
    sum = 0
    for y in Y:
        sum += nll_vel(K, b, sig_r, sig_y, t, y)
    return sum

def optimize_function(func, opt_vars: OrderedDict, fixed_vars: dict, opt_bounds=None, method=None, **kwargs):
    """
    Generalize the optimization of a function with respect to a subset of variables using named variables and fixed vectors.

    Parameters:
    :param func: The original function to optimize.
    :param opt_vars: Ordered dictionary of optimization variable names and their initial values.
    :param fixed_vars: Dict of fixed variable names and their values.
    :param opt_bounds: Bounds for the optimization variables [(min, max), ...].
    :param method: Optimization method.
    :param kwargs: Keyword arguments to be passed to scipy's minimize.

    Returns:
    result: The result of the optimization.
    """
    def wrapper(opt_values):
        # combine fixed arguments and optimization arguments into argument list for func
        full_args = fixed_vars.copy()
        for idx, key in enumerate(opt_vars):
            full_args[key] = opt_values[idx]
        
        return func(**full_args)
    init_vals = np.asarray(list(opt_vars.values()))
    result = minimize(wrapper, init_vals, bounds=opt_bounds, method=method, **kwargs)
    return result


# %% [markdown]
# ## Data from multivariate Gaussian

# %%
def gen_data(t, mean_proc, kern, reps=1, seed=None):
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    mu = mean_proc(t, v=DEF_PARAMS['V'])
    sig = kern(t, t)
    Y = rng.multivariate_normal(mean=mu, cov=sig, size=reps)
    return Y


# %% [markdown]
# ### No noise
# Conclusion: easy, even with minimal samples and trials.

# %%
sig_r = 1e-10
sig_y = 1e-10

# %% [markdown]
# #### Infer $K$

# %%
opt_vars = OrderedDict([
    ('K', 0.15)
])
opt_bounds = [OPT_BOUNDS['K']]

# %%
# Position control
fixed_vars = {
    'sig_r': sig_r,
    'sig_y': sig_y,
    'gamma': DEF_PARAMS['gamma']
}
pos_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE))) * np.nan,
                       coords=dict(
                           n_samples=N_SAMPLES_RANGE,
                           n_trials=N_TRIALS_RANGE
                       ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    t = np.linspace(0, T_MAX, n_samples)
    fixed_vars['t'] = t
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        mean_proc = PosControlMeanProcess(DEF_PARAMS['K'])
        kern = PosControlKernel(DEF_PARAMS['K'], sig_r, sig_y, DEF_PARAMS['gamma'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_pos, opt_vars, fixed_vars, opt_bounds)
        pos_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x[0]
        if np.isclose(res.x[0], DEF_PARAMS['K']):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K']):
            break
print('Done')

# %%
pos_results.values

# %%
# Velocity control
fixed_vars = {
    'sig_r': sig_r,
    'sig_y': sig_y,
    'b': DEF_PARAMS['b']
}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE))) * np.nan,
                          coords=dict(
                            n_samples=N_SAMPLES_RANGE,
                            n_trials=N_TRIALS_RANGE
                          ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    t = np.linspace(0, T_MAX, n_samples)
    fixed_vars['t'] = t
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], sig_r, sig_y)
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x[0]
        if np.isclose(res.x[0], DEF_PARAMS['K']):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K']):
            break
print('Done')

# %%
vel_results.values

# %% [markdown]
# ### Perceptual noise
# Conclusion: still easy for the position control with minimal samples and trials. Velocity control is harder, requires more than one trial to get close to the true K value, and when trying K and $\sigma_r$ together, takes 1000 trials.

# %%
sig_y = 1e-10

# %% [markdown]
# #### Infer $K$

# %%
opt_vars = OrderedDict([
    ('K', 0.15),
])
opt_bounds = [OPT_BOUNDS['K']]

# %%
# Position control
fixed_vars = {
    'sig_r': DEF_PARAMS['sig_r'],
    'sig_y': sig_y,
    'gamma': DEF_PARAMS['gamma']
}
pos_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE))) * np.nan,
                          coords=dict(
                            n_samples=N_SAMPLES_RANGE,
                            n_trials=N_TRIALS_RANGE
                          ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    t = np.linspace(0, T_MAX, n_samples)
    fixed_vars['t'] = t
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        mean_proc = PosControlMeanProcess(DEF_PARAMS['K'])
        kern = PosControlKernel(DEF_PARAMS['K'], DEF_PARAMS['sig_r'], sig_y, DEF_PARAMS['gamma'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_pos, opt_vars, fixed_vars, opt_bounds)
        pos_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x[0]
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
print('Done')

# %%
pos_results.values

# %%
# Velocity control
fixed_vars = {
    'sig_r': DEF_PARAMS['sig_r'],
    'sig_y': sig_y,
    'b': DEF_PARAMS['b']
}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE
                            ))
for n_trials in tqdm(N_TRIALS_RANGE, position=0):
    for n_samples in tqdm(N_SAMPLES_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], DEF_PARAMS['sig_r'], sig_y)
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x[0]
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=5*1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=5*1e-2):
            break
print('Done')

# %%
print(f'finished at n_samples={n_samples}, n_trials={n_trials}')
vel_results.values

# %% [markdown]
# #### Infer $K$, $\sigma_r$

# %%
opt_vars = OrderedDict([
    ('K', 0.15),
    ('sig_r', 0.01)
])
opt_bounds = [OPT_BOUNDS['K'], OPT_BOUNDS['sig_r']]

# %%
# Position control
fixed_vars = {
    'sig_y': sig_y,
    'gamma': DEF_PARAMS['gamma']
}
pos_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                          coords=dict(
                              n_samples=N_SAMPLES_RANGE,
                              n_trials=N_TRIALS_RANGE,
                              opt_vars=list(opt_vars.keys())
                          ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    t = np.linspace(0, T_MAX, n_samples)
    fixed_vars['t'] = t
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        mean_proc = PosControlMeanProcess(DEF_PARAMS['K'])
        kern = PosControlKernel(DEF_PARAMS['K'], DEF_PARAMS['sig_r'], sig_y, DEF_PARAMS['gamma'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_pos, opt_vars, fixed_vars, opt_bounds)
        pos_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2):
            break
print('Done')

# %%
print('K:')
print(pos_results.sel(opt_vars='K').values)
print('sig_r:')
print(pos_results.sel(opt_vars='sig_r').values)

# %%
# Velocity control
fixed_vars = {
    'sig_y': sig_y,
    'b': DEF_PARAMS['b']
}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                          coords=dict(
                              n_samples=N_SAMPLES_RANGE,
                              n_trials=N_TRIALS_RANGE,
                              opt_vars=list(opt_vars.keys())
                          ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], DEF_PARAMS['sig_r'], sig_y)
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2):
            break
print('Done')

# %%
print(f'finished at n_samples={n_samples}, n_trials={n_trials}')
print('K:')
print(vel_results.sel(opt_vars='K').values)
print('sig_r:')
print(vel_results.sel(opt_vars='sig_r').values)

# %% [markdown]
# ### Measurement noise
# Conclusion: when inferring only K, easy for both with minimal samples and trials. When inferring K and $\sigma_y$, both models require more than minimal trials and samples. Velocity does worse, requires 1000 trials.

# %%
sig_r = 1e-10

# %% [markdown]
# #### Infer $K$

# %%
opt_vars = OrderedDict([
    ('K', 0.15),
])
opt_bounds = [OPT_BOUNDS['K']]

# %%
# Position control
fixed_vars = {
    'sig_r': sig_r,
    'sig_y': DEF_PARAMS['sig_y'],
    'gamma': DEF_PARAMS['gamma']
}
pos_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    t = np.linspace(0, T_MAX, n_samples)
    fixed_vars['t'] = t
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        mean_proc = PosControlMeanProcess(DEF_PARAMS['K'])
        kern = PosControlKernel(DEF_PARAMS['K'], sig_r, DEF_PARAMS['sig_y'], DEF_PARAMS['gamma'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_pos, opt_vars, fixed_vars, opt_bounds)
        pos_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x[0]
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
print('Done')

# %%
pos_results.values

# %%
# Velocity control
fixed_vars = {
    'sig_r': sig_r,
    'sig_y': DEF_PARAMS['sig_y'],
    'b': DEF_PARAMS['b']
}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    t = np.linspace(0, T_MAX, n_samples)
    fixed_vars['t'] = t
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], sig_r, DEF_PARAMS['sig_y'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x[0]
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
print('Done')

# %%
vel_results.values

# %% [markdown]
# #### Infer $K$, $\sigma_y$

# %%
opt_vars = OrderedDict([
    ('K', 0.15),
    ('sig_y', 0.01)
])
opt_bounds = [OPT_BOUNDS['K'], OPT_BOUNDS['sig_y']]

# %%
# Position control
fixed_vars = {
    'sig_r': sig_r,
    'gamma': DEF_PARAMS['gamma']
}
pos_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE,
                                opt_vars=list(opt_vars.keys())
                            ))
for n_trials in tqdm(N_TRIALS_RANGE, position=0):
    for n_samples in tqdm(N_SAMPLES_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = PosControlMeanProcess(DEF_PARAMS['K'])
        kern = PosControlKernel(DEF_PARAMS['K'], sig_r, DEF_PARAMS['sig_y'], DEF_PARAMS['gamma'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_pos, opt_vars, fixed_vars, opt_bounds)
        pos_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
print('Done')

# %%
print('K:')
print(pos_results.sel(opt_vars='K').values)
print('sig_y:')
print(pos_results.sel(opt_vars='sig_y').values)

# %%
# Velocity control
fixed_vars = {
    'sig_r': sig_r,
    'b': DEF_PARAMS['b']
}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE,
                                opt_vars=list(opt_vars.keys())
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], sig_r, DEF_PARAMS['sig_y'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
print('Done')

# %%
print('K:')
print(vel_results.sel(opt_vars='K').values)
print('sig_y:')
print(vel_results.sel(opt_vars='sig_y').values)

# %% [markdown]
# ### Perceptual + Measurement noise
# Conclusion: now, even inferring K alone is hard for the velocity model, requiring a lot (but not maximal) amount of data. Adding $\sigma_r$ to the mix makes it even harder, requiring more data for the position model, and 1000 trials with 50 samples each for the velocity model. Inferring $\sigma_y$ is worse - requires 1000 trials from both. Inferring all three parameters is not harder than just K and $\sigma_y$.

# %% [markdown]
# #### Infer $K$

# %%
opt_vars = OrderedDict([
    ('K', 0.15),
])
opt_bounds = [OPT_BOUNDS['K']]

# %%
# Position control
fixed_vars = {
    'sig_r': DEF_PARAMS['sig_r'],
    'sig_y': DEF_PARAMS['sig_y'],
    'gamma': DEF_PARAMS['gamma']
}
pos_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    t = np.linspace(0, T_MAX, n_samples)
    fixed_vars['t'] = t
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        mean_proc = PosControlMeanProcess(DEF_PARAMS['K'])
        kern = PosControlKernel(DEF_PARAMS['K'], DEF_PARAMS['sig_r'], DEF_PARAMS['sig_y'], DEF_PARAMS['gamma'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_pos, opt_vars, fixed_vars, opt_bounds)
        pos_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x[0]
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
print('Done')

# %%
pos_results.values

# %%
# Velocity control
fixed_vars = {
    'sig_r': DEF_PARAMS['sig_r'],
    'sig_y': DEF_PARAMS['sig_y'],
    'b': DEF_PARAMS['b']
}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE
                            ))
for n_trials in tqdm(N_TRIALS_RANGE, position=0):
    for n_samples in tqdm(N_SAMPLES_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], DEF_PARAMS['sig_r'], DEF_PARAMS['sig_y'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x[0]
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2):
            break
print('Done')

# %%
vel_results.values

# %% [markdown]
# #### Infer $K$, $\sigma_r$

# %%
opt_vars = OrderedDict([
    ('K', 0.15),
    ('sig_r', 0.01)
])
opt_bounds = [OPT_BOUNDS['K'], OPT_BOUNDS['sig_r']]

# %%
# Position control
fixed_vars = {
    'sig_y': DEF_PARAMS['sig_y'],
    'gamma': DEF_PARAMS['gamma']
}
pos_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE,
                                opt_vars=list(opt_vars.keys())
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = PosControlMeanProcess(DEF_PARAMS['K'])
        kern = PosControlKernel(DEF_PARAMS['K'], DEF_PARAMS['sig_r'], DEF_PARAMS['sig_y'], DEF_PARAMS['gamma'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_pos, opt_vars, fixed_vars, opt_bounds)
        pos_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2):
            break
print('Done')

# %%
print(f'finished at n_samples={n_samples}, n_trials={n_trials}')
print('K:')
print(pos_results.sel(opt_vars='K').values)
print('sig_r:')
print(pos_results.sel(opt_vars='sig_r').values)

# %%
# Velocity control
fixed_vars = {
    'sig_y': DEF_PARAMS['sig_y'],
    'b': DEF_PARAMS['b']
}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE,
                                opt_vars=list(opt_vars.keys())
                            ))
for n_trials in tqdm(N_TRIALS_RANGE, position=0):
    for n_samples in tqdm(N_SAMPLES_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], DEF_PARAMS['sig_r'], DEF_PARAMS['sig_y'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2):
            break
print('Done')

# %%
print(f'finished at n_samples={n_samples}, n_trials={n_trials}')
print('K:')
print(vel_results.sel(opt_vars='K').values)
print('sig_r:')
print(vel_results.sel(opt_vars='sig_r').values)

# %% [markdown]
# #### Infer $K$, $\sigma_y$

# %%
opt_vars = OrderedDict([
    ('K', 0.15),
    ('sig_y', 0.01)
])
opt_bounds = [OPT_BOUNDS['K'], OPT_BOUNDS['sig_y']]

# %%
# Position control
fixed_vars = {
    'sig_r': DEF_PARAMS['sig_r'],
    'gamma': DEF_PARAMS['gamma']
}
pos_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE,
                                opt_vars=list(opt_vars.keys())
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = PosControlMeanProcess(DEF_PARAMS['K'])
        kern = PosControlKernel(DEF_PARAMS['K'], DEF_PARAMS['sig_r'], DEF_PARAMS['sig_y'], DEF_PARAMS['gamma'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_pos, opt_vars, fixed_vars, opt_bounds)
        pos_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
print('Done')

# %%
print(f'finished at n_samples={n_samples}, n_trials={n_trials}')
print('K:')
print(pos_results.sel(opt_vars='K').values)
print('sig_y:')
print(pos_results.sel(opt_vars='sig_y').values)

# %%
# Velocity control
fixed_vars = {
    'sig_r': DEF_PARAMS['sig_r'],
    'b': DEF_PARAMS['b']
}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE,
                                opt_vars=list(opt_vars.keys())
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], DEF_PARAMS['sig_r'], DEF_PARAMS['sig_y'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
print('Done')

# %%
print(f'finished at n_samples={n_samples}, n_trials={n_trials}')
print('K:')
print(vel_results.sel(opt_vars='K').values)
print('sig_y:')
print(vel_results.sel(opt_vars='sig_y').values)

# %% [markdown]
# #### Infer $K$, $\sigma_r$, $\sigma_y$

# %%
opt_vars = OrderedDict([
    ('K', 0.15),
    ('sig_r', 0.01),
    ('sig_y', 0.01)
])
opt_bounds = [OPT_BOUNDS['K'], OPT_BOUNDS['sig_r'], OPT_BOUNDS['sig_y']]

# %%
# Position control
fixed_vars = {
    'gamma': DEF_PARAMS['gamma']
}
pos_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE,
                                opt_vars=list(opt_vars.keys())
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = PosControlMeanProcess(DEF_PARAMS['K'])
        kern = PosControlKernel(DEF_PARAMS['K'], DEF_PARAMS['sig_r'], DEF_PARAMS['sig_y'], DEF_PARAMS['gamma'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_pos, opt_vars, fixed_vars, opt_bounds)
        pos_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2) and np.isclose(res.x[2], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2) and np.isclose(res.x[2], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
print('Done')

# %%
print(f'finished at n_samples={n_samples}, n_trials={n_trials}')
print('K:')
print(pos_results.sel(opt_vars='K').values)
print('sig_r:')
print(pos_results.sel(opt_vars='sig_r').values)
print('sig_y:')
print(pos_results.sel(opt_vars='sig_y').values)

# %%
# Velocity control
fixed_vars = {
    'b': DEF_PARAMS['b']
}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE,
                                opt_vars=list(opt_vars.keys())
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], DEF_PARAMS['sig_r'], DEF_PARAMS['sig_y'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2) and np.isclose(res.x[2], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2) and np.isclose(res.x[2], DEF_PARAMS['sig_y'], rtol=1e-2):
            break
print('Done')

# %%
print(f'finished at n_samples={n_samples}, n_trials={n_trials}')
print('K:')
print(vel_results.sel(opt_vars='K').values)
print('sig_r:')
print(vel_results.sel(opt_vars='sig_r').values)

# %% [markdown]
# #### Infer $K$, $\sigma_r$, $\sigma_y$, $b$  
# Currently not working, during optimization, the kernel matrix becomes non-positive definite, despite the bounds on the parameters.

# %%
opt_vars = OrderedDict([
    ('K', 0.15),
    ('sig_r', 0.01),
    ('sig_y', 0.01),
    ('b', 0.01)
])
opt_bounds = [OPT_BOUNDS['K'], OPT_BOUNDS['sig_r'], OPT_BOUNDS['sig_y'], OPT_BOUNDS['b']]

# %%
# Velocity control
fixed_vars = {}
vel_results = xr.DataArray(data=np.zeros(shape=(len(N_SAMPLES_RANGE), len(N_TRIALS_RANGE), len(opt_vars))) * np.nan,
                            coords=dict(
                                n_samples=N_SAMPLES_RANGE,
                                n_trials=N_TRIALS_RANGE,
                                opt_vars=list(opt_vars.keys())
                            ))
for n_samples in tqdm(N_SAMPLES_RANGE, position=0):
    for n_trials in tqdm(N_TRIALS_RANGE, position=1, leave=False):
        t = np.linspace(0, T_MAX, n_samples)
        fixed_vars['t'] = t
        mean_proc = VelControlMeanProcess(DEF_PARAMS['K'], DEF_PARAMS['b'])
        kern = VelControlKernel(DEF_PARAMS['K'], DEF_PARAMS['b'], DEF_PARAMS['sig_r'], DEF_PARAMS['sig_y'])
        Y = gen_data(t, mean_proc, kern, reps=n_trials)
        fixed_vars['Y'] = Y
        res = optimize_function(nll_sum_vel, opt_vars, fixed_vars, opt_bounds)
        vel_results.loc[dict(n_samples=n_samples, n_trials=n_trials)] = res.x
        if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2) and np.isclose(res.x[2], DEF_PARAMS['sig_y'], rtol=1e-2) and np.isclose(res.x[3], DEF_PARAMS['b'], rtol=1e-2):
            break
    if np.isclose(res.x[0], DEF_PARAMS['K'], rtol=1e-2) and np.isclose(res.x[1], DEF_PARAMS['sig_r'], rtol=1e-2) and np.isclose(res.x[2], DEF_PARAMS['sig_y'], rtol=1e-2) and np.isclose(res.x[3], DEF_PARAMS['b'], rtol=1e-2):
            break
print('Done')
