# A demo script for the DataAnalyzer class.
# Run this to demonstrate how the eyetracking and response data looks like.

from analysis.DataAnalyzer import DataAnalyzer
from util.motion import constant1d
import matplotlib.pyplot as plt

demo_path = r"/path/to/your/demo/data"  # Replace with the path to your demo data
da = DataAnalyzer(demo_path, demo_path)
da.prepare_et_data(force=False)
da.load_data(excluded=[], test=False)
da.calc_v()
part = da.participants[0]
trials = da.data[part]['et_trials']
results = da.data[part]['results']
params = da.data[part]['params']

# Plot eye-tracking data
for speed in results['speed'].sort_values().unique():
    fig, ax = plt.subplots(1)
    for ii in results[(results['speed'] == speed)]['thisN']:
        trial = trials[trials['this_n'] == ii]
        if len(trial.dropna()) == 0:
            continue
        this_result = results[results['thisN'] == ii]
        zeroed_time = trial['timestamp_ms'] - trial['timestamp_ms'].iloc[0]
        color = 'C0' if this_result['motion_left_to_right'].iloc[0] else 'C1'
        ax.plot(zeroed_time, trial['x_pix'], alpha=0.8, label=ii, c=color)
    x_0_1 = constant1d(zeroed_time / 1000, params['monitor_size_pix'][0] / 2 - params['motion_span_pix'] / 2, speed)
    x_0_2 = constant1d(zeroed_time / 1000, params['monitor_size_pix'][0] / 2 + params['motion_span_pix'] / 2, -speed)
    plt.plot(zeroed_time, x_0_1, 'k--')
    plt.plot(zeroed_time, x_0_2, 'k--')
    plt.title(f'All trials, speed={speed}')
    plt.xlabel('time (ms)')
    plt.ylabel('x position (px)')
    plt.show()

# Plot responses over time
for speed in results['speed'].sort_values().unique():
    fig, ax = plt.subplots(1)
    da.plot_trial_sequence(part, speed, ax=ax)
    ax.legend_.remove()
    plt.show()
