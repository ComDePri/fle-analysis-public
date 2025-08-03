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
# # Demo trial segmentation process
# Trials contain timepoints without tracking. We need to segment trials into epochs that contain only tracking moments. Given a trial data, we do it in the following steps:
# 1. Load trial data
# 2. Throw away open-loop timepoints
# 3. Throw away timepoints after the flasher appears
# 4. Mark timepoints with velocity over saccade threshold for exclusion. This identifies saccades and also noisy moments.
# 5. Inflate saccades to include surrounding timepoints. This is to avoid including saccade start/end in tracking epochs.
# 6. Mark timepoints with nan `x` or `y` values for exclusion. This identifies blinks.
# 7. Mark timepoints part of too-short unmarked timepoint sequences for exclusion.
# 8. Take all unmarked timepoints and segment them into epochs. These epochs are the suspected tracking epochs.
# 9. Fit epochs to a line, and exclude epochs with slope below threshold. These are epochs with no tracking.
# 10. Return the remaining epochs as tracking data.

# %%
from matplotlib import pyplot as plt
from util.et_utils import trim_ramping_time, trim_after_event, remove_short_segments, fit_seg, split, inflate_true_segments, identify_tracking_epochs
from analysis.DataAnalyzer import DataAnalyzer
from ComDePy.viz import set_defaults
from util.motion import constant1d
from util.FLE_utils import reflect

set_defaults()

# %% [markdown]
# ## Load trial data

# %%
da = DataAnalyzer(r"\\132.64.186.144\HartLabNAS\Experiments\FLE_exp3_eyetracker-experiment-2024",
                  r"\\132.64.186.144\HartLabNAS\Projects\FLE\exp3_et")
da.load_data(excluded=[], test=True)

# %%
part = da.participants[1]
n_trial = 131

twoAFCres = da.data[part]['results']
result = twoAFCres[twoAFCres['thisN'] == n_trial].iloc[0]
ltr = result['motion_left_to_right']
speed = result['speed']

params = da.data[part]['params']
screen_center = params['monitor_size_pix'][0] / 2

trials = da.data[part]['et_trials']
trial = trials[trials['this_n'] == n_trial].copy()
trial['trial_time'] = da.trial_time(part, n_trial)
trial['x_corrected'] = trial['x_pix'] if ltr else reflect(trial['x_pix'], screen_center)


# %%
def mover_x_t(t, speed):
    mover_start_offset = -params['motion_span_pix'] / 2
    mover_x_0 = params['monitor_size_pix'][0] / 2 + mover_start_offset
    return constant1d(t, mover_x_0, speed)

def plot_trial(trial, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(trial['trial_time'], trial['x_corrected'], label='x', lw=1)
    ax.plot(trial['trial_time'], mover_x_t(trial['trial_time'] / 1000, speed), label='mover', c='k', zorder=-10, lw=1)
    ax.set_title(f"Trial {n_trial}")
    ax.set_ylabel('x')
    ax.set_xlabel('time (ms)')
    ax.set_ylim([0, params['monitor_size_pix'][0]])
    ax.set_xlim(trial['trial_time'].min(), trial['trial_time'].max())
    ax.legend()
    return ax
    
def overlay_flasher(ax, trial):
    flasher_time = trial[trial['flasher']]['trial_time'].min()
    ax.axvline(flasher_time, color='k', linestyle='--', label='flasher', lw=1)
    ax.legend()
    return ax

def shade_bg(ax, t, marker_series, color='gray', alpha=0.1):
    # Shade background according to True values in marker_series
    # get y lims
    ymin, ymax = ax.get_ylim()
    ax.fill_between(t, ymin, ymax, where=marker_series.values, color=color, alpha=alpha)


# %%
# Plot the trial
ax = plot_trial(trial)
overlay_flasher(ax, trial)

# %% [markdown]
# ## Trim open-loop timepoints

# %%
ramp_time = 200
trimmed_trial = trim_ramping_time(trial, ramp_time)
ax = plot_trial(trial)
overlay_flasher(ax, trial)
ax.plot(trimmed_trial['trial_time'], trimmed_trial['x_corrected'], label='trimmed x', c='r')
ax.axvline(trimmed_trial['trial_time'].min(), color='C2', linestyle='--', label='end of ramping')
ax.legend()
ax.set_title('Trimmed after ramping')

# %% [markdown]
# ## Trim post-flash timepoints

# %%
trimmed_trial = trim_after_event(trimmed_trial, 'flasher')
ax = plot_trial(trial)
overlay_flasher(ax, trial)
ax.plot(trimmed_trial['trial_time'], trimmed_trial['x_corrected'], label='trimmed x', c='r')
ax.set_title('Trimmed before flasher')

# %% [markdown]
# ## Mark potential saccades

# %%
saccade_speed_thresh = 2000
window = 11
trimmed_trial['v_x'] = trimmed_trial['x_corrected'].diff() / trimmed_trial['trial_time'].diff() * 1000
trimmed_trial['v_x_smooth'] = trimmed_trial['v_x'].rolling(window=window, center=True).mean()
trimmed_trial['saccade'] = trimmed_trial['v_x_smooth'].abs() > saccade_speed_thresh


# %%
# Plot speed and position in two subplots, with saccades shaded
fig, axs = plt.subplots(2, 1, sharex=True)
plot_trial(trial, ax=axs[0])
shade_bg(axs[0], trimmed_trial['trial_time'], trimmed_trial['saccade'])
axs[0].set_title('Position')
axs[0].set_xlabel('')
overlay_flasher(axs[0], trial)
shade_bg(axs[0], trimmed_trial['trial_time'], trimmed_trial['saccade'])
axs[0].get_legend().remove()
axs[0].set_xlim(trimmed_trial['trial_time'].min(), trimmed_trial[trimmed_trial['flasher']]['trial_time'].min())


axs[1].plot(trimmed_trial['trial_time'], trimmed_trial['v_x'], label='v_x', lw=1)
axs[1].plot(trimmed_trial['trial_time'], trimmed_trial['v_x_smooth'], label='v_x_smooth', lw=0.8)
axs[1].axhline(saccade_speed_thresh, color='k', linestyle='--', label='saccade threshold', lw=0.5)
axs[1].axhline(-saccade_speed_thresh, color='k', linestyle='--', label='saccade threshold', lw=0.5)
overlay_flasher(axs[1], trial)
shade_bg(axs[1], trimmed_trial['trial_time'], trimmed_trial['saccade'])
axs[1].set_title('Speed')
axs[1].set_xlabel('time (ms)')
axs[1].get_legend().remove()
axs[1].set_xlim(trimmed_trial['trial_time'].min(), trimmed_trial[trimmed_trial['flasher']]['trial_time'].min())
axs[1].set_ylim(-25000, 40000)

# %% [markdown]
# ## Inflate saccades    

# %%
inflation = 15
trimmed_trial['saccade_inf'] = inflate_true_segments(trimmed_trial['saccade'], inflation)

# %%
# Plot position with saccades and inflated saccades shaded
ax = plot_trial(trial)
overlay_flasher(ax, trial)
shade_bg(ax, trimmed_trial['trial_time'], trimmed_trial['saccade'])
shade_bg(ax, trimmed_trial['trial_time'], trimmed_trial['saccade_inf'] & ~trimmed_trial['saccade'], color='C2', alpha=0.3)
ax.set_xlim(trimmed_trial['trial_time'].min(), trimmed_trial[trimmed_trial['flasher']]['trial_time'].min())
ax.set_title('Saccades and inf saccades')
ax.get_legend().remove()

# %% [markdown]
# ## Mark blinks

# %%
trimmed_trial['blink'] = trimmed_trial['x_corrected'].isna() | trimmed_trial['y_pix'].isna()

# %%
# Plot position with blinks shaded
ax = plot_trial(trial)
overlay_flasher(ax, trial)
shade_bg(ax, trimmed_trial['trial_time'], trimmed_trial['blink'])
ax.set_title('Position, blinks highlighted')

# %% [markdown]
# ## Mark short segments

# %%
min_segment_length = 50
blink_or_saccade = trimmed_trial['blink'] | trimmed_trial['saccade_inf']
not_blink_or_saccade = ~blink_or_saccade
short_segments_removed = remove_short_segments(not_blink_or_saccade, min_segment_length)
trimmed_trial['suspected_tracking'] = short_segments_removed

# %%
# Plot position with short segments shaded
ax = plot_trial(trial)
overlay_flasher(ax, trial)
# highlight everything in not_blink_or_saccade but not in short_segments_removed
shade_bg(ax, trimmed_trial['trial_time'], not_blink_or_saccade & ~short_segments_removed)
ax.set_title('Position, short segments highlighted')

# %% [markdown]
# ## Segment epochs

# %%
epochs = split(trimmed_trial, 'suspected_tracking')

# %%
# Plot position with epochs shaded
ax = plot_trial(trial)
overlay_flasher(ax, trial)
for epoch in epochs:
    shade_bg(ax, epoch['trial_time'], epoch['suspected_tracking'])
ax.set_title('Suspected tracking epochs')

# %% [markdown]
# ## Fit epochs

# %%
fits = [fit_seg(epoch['trial_time'] / 1000, epoch['x_corrected']) for epoch in epochs]

# %%
# Plot position with fits
ax = plot_trial(trial)
overlay_flasher(ax, trial)
for epoch, fit in zip(epochs, fits):
    fit_x = fit[0] * epoch['trial_time'] / 1000 + fit[1]
    ax.plot(epoch['trial_time'], fit_x, lw=2, c='C2', ls='--', alpha=0.8)
ax.set_title('Position, fits overlaid')

# %% [markdown]
# ## Return tracking data

# %%
slope_thresh = 0.25 * speed
tracking_epochs = [epoch for epoch, fit in zip(epochs, fits) if fit[0] > slope_thresh]

# %%
# Plot position with tracking epochs shaded
ax = plot_trial(trial)
overlay_flasher(ax, trial)
for epoch in tracking_epochs:
    shade_bg(ax, epoch['trial_time'], epoch['suspected_tracking'])
ax.set_title('Position, tracking epochs highlighted')

# %% [markdown]
# ## All in one

# %%
trial = trials[trials['this_n'] == n_trial].copy()
trial['trial_time'] = da.trial_time(part, n_trial)
trial['x_corrected'] = trial['x_pix'] if ltr else reflect(trial['x_pix'], screen_center)
trimmed_trial = trim_ramping_time(trial, ramp_time)
trimmed_trial = trim_after_event(trimmed_trial, 'flasher')
trimmed_trial['v_x'] = trimmed_trial['x_corrected'].diff() / trimmed_trial['trial_time'].diff() * 1000
trimmed_trial['v_x_smooth'] = trimmed_trial['v_x'].rolling(window=window, center=True).mean()
trimmed_trial['saccade'] = trimmed_trial['v_x_smooth'].abs() > saccade_speed_thresh
trimmed_trial['saccade_inf'] = inflate_true_segments(trimmed_trial['saccade'], inflation)
trimmed_trial['blink'] = trimmed_trial['x_corrected'].isna() | trimmed_trial['y_pix'].isna()
blink_or_saccade = trimmed_trial['blink'] | trimmed_trial['saccade_inf']
not_blink_or_saccade = ~blink_or_saccade
short_segments_removed = remove_short_segments(not_blink_or_saccade, min_segment_length)
trimmed_trial['suspected_tracking'] = short_segments_removed
epochs = split(trimmed_trial, 'suspected_tracking')
fits = [fit_seg(epoch['trial_time'] / 1000, epoch['x_corrected']) for epoch in epochs]
slope_thresh = 0.25 * speed
tracking_epochs = [epoch for epoch, fit in zip(epochs, fits) if fit[0] > slope_thresh]

print(f"Trial {n_trial} segmented into {len(tracking_epochs)} tracking epochs")
print("start", "end")
for epoch in tracking_epochs:
    print(epoch['trial_time'].min(), epoch['trial_time'].max())

# %% [markdown]
# ### Using the function

# %%
tracking_epochs = identify_tracking_epochs(trial, slope_thresh=slope_thresh, ramp_time=ramp_time, window=window, v_thresh=saccade_speed_thresh, inflation=inflation, min_length=min_segment_length)
print(f"Trial {n_trial} segmented into {len(tracking_epochs)} tracking epochs")
print("start", "end")
for epoch in tracking_epochs:
    print(epoch['trial_time'].min(), epoch['trial_time'].max())
