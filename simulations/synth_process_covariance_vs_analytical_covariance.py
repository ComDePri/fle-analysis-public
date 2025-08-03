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
# # Test covariance agreement between analytical kernel and synthetic process
#  
# Check whether covariance matrices calculated from synthetic data and analytical kernel are similar, and converge with increasing number of trajectories.
#
# If the analytical expression is correct (and the process implementation is correct), the covariance matrices calculated from the kernel and synthetic data should be similar. The covariance matrices calculated from the synthetic data should converge to the kernel as the number of trajectories increases.
#
# Interim conclusion (2024-06-27): there seems to be an off-by-one timepoint difference between velocity process and kernel, unclear what source, but probably won't be a problem in real-world scenario. Position kernel matches synthetic process well, up to small bias in the beginning, probably resulting from Euler-Maruyama approximation being 1st order, and the continuous-time mean solution being convex.

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from analysis.bayesian_inference_et import PosControlKernel, PosControlMeanProcess, VelControlKernel, VelControlMeanProcess
from simulations.synthetic_et_process import PosControlAgent, VelControlAgent
from ComDePy.numerical import MultiAgent
from ComDePy.viz import set_defaults, tile_subplots
set_defaults()

# decreaese default fig size and font size
mpl.rcParams['figure.figsize'] = [6.0, 4.0]
mpl.rcParams['font.size'] = 6

# %% [markdown]
# ## Position control

# %%
# Parameters
T_max = 100
t = np.arange(T_max)
Ns = [10, 100, 1000]
K = 0.5
sig_r = 0.01
sig_y = 0.01
V = 0.5

def pos_noise(rng, x, t):
    return rng.normal(0, sig_r)

def pos_obs_noise(rng, x, t):
    return rng.normal(0, sig_y)

def target(t):
    return V * t

# Kernel
kernel = PosControlKernel(K, sig_r, sig_y, gamma=0)

# Agent ctor
agent_ctor = lambda: PosControlAgent(K=K, x0=0, target=target, step_noise=pos_noise, meas_noise=pos_obs_noise, max_steps=T_max)

def get_agent(agent):
    return agent



# %%
# Calculate kernel
kern = kernel(t, t)
# define colorbar based on kernel's range
vmax = np.max(np.abs(kern))
vmin = 0
# Generate synthetic data
mu = PosControlMeanProcess(K)(t, v=V)
means_pos = []
covs_pos = []
covs_emp_pos = []
for ii, n in enumerate(Ns):
    ma = MultiAgent(n, agent_ctor, get_agent)
    ma.sim()
    mean = np.zeros(T_max)
    cov = np.zeros((T_max, T_max))
    for agent in ma.get_data():
        x = agent.x_meas
        mean += x
        x_tilde = x - mu
        this_cov = np.outer(x_tilde, x_tilde)
        cov += this_cov
    mean /= n
    means_pos.append(mean)
    cov /= n
    covs_pos.append(cov)
    cov_emp = np.zeros((T_max, T_max))
    for agent in ma.get_data():
        x_tilde_emp = agent.x_meas - mean
        this_cov_emp = np.outer(x_tilde_emp, x_tilde_emp)
        cov_emp += this_cov_emp
    cov_emp /= n
    covs_emp_pos.append(cov_emp)

# %%
fig_mean, ax_mean = plt.subplots(1,2)
for ii, (n, mean) in enumerate(zip(Ns, means_pos)):
    ax_mean[0].plot(t, mean, label=fr"Data mean, $\bar{{Y}}, n={n}$", lw=1)
    ax_mean[1].plot(t, mu - mean, label=fr"Data residual, $\mu - \bar{{Y}}$, n={n}", lw=1)
ax_mean[0].plot(t, mu, c='k', label=f"Expected mean, $\mu$", lw=1)
ax_mean[0].plot(t, target(t), c='k', ls='--', label=f"Target, $Vt$", lw=1)
ax_mean[0].legend(fontsize=6)
fig_mean.supxlabel("Time")
ax_mean[0].set_ylabel("Position (residual)")
ax_mean[0].set_title("Mean of synthetic data")
ax_mean[1].set_title("Residual of synthetic data")
ax_mean[1].legend(fontsize=6)
ax_mean[1].axhline(0, c='k', ls='--', lw=1)

fig_cov, ax_cov = tile_subplots(len(Ns) + 1)
ax_cov = ax_cov[:len(Ns) + 1]
ax_cov[-1].imshow(kern, vmin=vmin, vmax=vmax)
ax_cov[-1].set_title("Kernel")
ax_cov[-1].invert_yaxis()
ax_cov[-1].grid(False)
ax_cov[-1].set_xlabel("Time")
ax_cov[-1].set_ylabel("Time")

for ii, (n, cov) in enumerate(zip(Ns, covs_pos)):
    ax_cov[ii].imshow(cov, vmin=vmin, vmax=vmax)
    ax_cov[ii].set_title(f"n={n}")
    # fig_cov.colorbar(ax_cov[ii].imshow(cov), ax=ax_cov[ii])
    ax_cov[ii].invert_yaxis()
    ax_cov[ii].grid(False)
    ax_cov[ii].set_xlabel("Time")
    ax_cov[ii].set_ylabel("Time")
    
fig_cov.suptitle("Covariance of synthetic data")

# %% [markdown]
# ## Velocity control

# %%
# Parameters
T_max = 100
t = np.arange(T_max)
Ns = [10, 100, 1000]
K = 0.5
b = 0.1
sig_r = 0.01
sig_y = 0.01
V = 1

def vel_noise(rng, x, t):
    return rng.normal(0, sig_r)

def vel_obs_noise(rng, x, t):
    return rng.normal(0, sig_y)

def target(t):
    return V

# Kernel
kernel = VelControlKernel(K, b, sig_r, sig_y)

# Agent ctor
agent_ctor = lambda: VelControlAgent(K=K, b=b, v0=0, target=target, step_noise=vel_noise, meas_noise=vel_obs_noise, max_steps=T_max)

def get_agent(agent):
    return agent



# %%
# Calculate kernel
kern = kernel(t, t)
# Generate synthetic data
mu = VelControlMeanProcess(K, b)(t, v=V)

means_vel = []
covs_vel = []
covs_emp_vel = []
for ii, n in enumerate(Ns):
    ma = MultiAgent(n, agent_ctor, get_agent)
    ma.sim()
    mean = np.zeros(T_max)
    cov = np.zeros((T_max, T_max))
    for agent in ma.get_data():
        x = agent.x_meas
        mean += x
        x_tilde = x - mu
        this_cov = np.outer(x_tilde, x_tilde)
        cov += this_cov
    mean /= n
    means_vel.append(mean)
    
    cov /= n
    covs_vel.append(cov)
    
    cov_emp = np.zeros((T_max, T_max))
    for agent in ma.get_data():
        x_tilde_emp = agent.x_meas - mean
        this_cov_emp = np.outer(x_tilde_emp, x_tilde_emp)
        cov_emp += this_cov_emp
    cov_emp /= n
    covs_emp_vel.append(cov_emp)

# %%
vmax = np.max(np.abs(kern))
vmin = 0

fig_mean, ax_mean = plt.subplots()
for ii, (n, mean) in enumerate(zip(Ns, means_vel)):
    ax_mean.plot(t[1:], mu[1:] - mean[:-1], label=fr"Data mean, $\mu-\bar{{Y}}$, n={n}")
ax_mean.axhline(0, 0, T_max, c='k', ls='--', lw=1)
ax_mean.legend()
ax_mean.set_xlabel("Time")
ax_mean.set_ylabel("Position (residual)")
fig_mean.suptitle("Mean of synthetic data")
   
fig_cov, ax_cov = tile_subplots(len(Ns) + 1)
ax_cov = ax_cov[:len(Ns) + 1]
im = ax_cov[-1].imshow(kern)
ax_cov[-1].set_title("Kernel")
ax_cov[-1].invert_yaxis()
ax_cov[-1].grid(False)
ax_cov[-1].set_xlabel("Time")
ax_cov[-1].set_ylabel("Time")

for ii, (n, cov) in enumerate(zip(Ns, covs_emp_vel)):
    ax_cov[ii].imshow(cov, vmin=vmin, vmax=vmax)
    ax_cov[ii].set_title(f"n={n}")
    ax_cov[ii].invert_yaxis()
    ax_cov[ii].grid(False)
    ax_cov[ii].set_xlabel("Time")
    ax_cov[ii].set_ylabel("Time")

fig_cov.suptitle("Covariance of synthetic data")
