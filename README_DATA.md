# Data README

This README explains the structure of the data files saved by running ExperimentDriver.py
The data files are saved in a `/data` directory. Each participant is assigned a unique ID, and their data files are saved in the participant subfolder `/data/{date}_{time}_{uid}`.

After a successful run, the subfolder has the following structure:

1. `params.json`: The parameters used for the experiment, including the demographic information.
2. `results.csv`: The raw results of the experiment, as well as some derived quantities calculated
   online.
3. `training.csv`: The results of the training phase of the experiment.
4. `attention_checks.csv`: The results of the attention checks in the experiment.
5. `extended_data/`: a subfolder containing the full prior and posterior distributions of the psychometric function
   parameters.
6. `{uid}.edf`: The eye tracking data, if eye tracking was enabled.

## `params.json`

This file contains a dictionary with two types of entries: data defining the experiment parameters,
and data defining the demographic information of the participant.

### Experiment parameters

Experiment structure parameters:

- `n_trials`: The number of trials in the experiment, for each condition.
- `speeds`: A list of speed conditions, in pixels/sec.
- `n_training_trials`: The number of trials in the training phase, for each condition - moving and stationary.
- `break_every`: The number of trials between breaks.
- `attention_check_every`: The number of trials between attention checks.
- `passes_to_skip`: The number of passes over the set of conditions to skip before starting to collect data. There is a
  need to skip passes to exclude the first few trials in each speed, since participants tend to respond randomly in the
  beginning of the experiment.
  Visual parameters:
- `flash_duration_ms`: The duration of the flash, in milliseconds. Actual flash duration is determined by the refresh
  rate of the monitor: `int(flash_duration_sec / frame_duration_sec)`.
  `frame_duration_sec` is calculated from `fps` below.
- `motion_span_pix`: The number of pixels the mover moves across in the screen in a trial.
- `max_offset_pix`: The maximum number of pixels the flasher can be offset from the mover at presentation time.
- `max_mover_pos_for_flash`: The maximum number of pixels the mover can be offset from the center of the screen at
  presentation time. This is equal to `int('motion_span_pix' / 2 - 'max_offset_pix')`.
- `vertical_stimuli_offset_pix`: The vertical offset between mover and flasher, in pixels. They are offset symmetrically
  from the center of the screen.
- `stim_color`: The color of the stimuli.
- `stim_radius_pix`: The radius of the stimuli, in pixels.
  Monitor parameters:
- `monitor_size_pix`: The size of the monitor, in pixels.
- `monitor_width_cm`: The width of the monitor, in centimeters. Used to calculate visual element sizes defined by their
  visual angle, along with `monitor_distance_cm`.
- `monitor_distance_cm`: The distance between the participant and the monitor, in centimeters. Used to calculate visual
  element sizes defined by their visual angle, along with `monitor_width_cm`. This value is generic and not measured for each run.
- `fps`: number of frames per second. Measured at run time.

### Demographic parameters

These parameters are collected from the participant at run time.

- `id`: A UUID integer of length 39, generated at run time. A truncated version appears in the filename.
- `age` Self-reported age in years.
- `gender`: Self-reported gender, out of the options "Prefer not to answer", "Woman", "Man", "Other".
- `education_years_complete`: Self-reported number of years of education completed.
- `time`: The time the experiment started, in the local time zone.

## `results.csv`

This is the main results file. This is a table with a row for each trial, with the following columns:

- `speed`: The speed condition of the trial. The speed of the mover in pixels/sec.
- `thisRepN`: What is the repetition number of this condition, i.e. what pass over the conditions.
  Goes from `1+passes_to_skip` to `n_trials`.
- `thisN`: The overall trial number. Goes from 0 to `(n_trials-passes_to_skip)*len(speeds)-1`.
- `intensity`: The offset of the stimulus in this trial, in pixels.
- `response`: The response of the participant in this trial, either `'left'` or `'right'`. This is the raw data.
- `response_time_ms`: The time it took the participant to respond, in milliseconds, **measured from flasher onset**.
  This is the raw data.
- `premature_responses`: The number of premature responses the participant made in this trial. This is the raw data.
  **Note**: the premature responses themselves are not logged and don't influence the sampling or the results.
- `exp_time_sec`: The overall time elapsed since the start of the experiment, in seconds. This is the raw data.
- `mean_est`: The estimated mean **parameter** of the participant's psychometric response function, in pixels
  (measured from zero offset), for the given speed condition, after the current trial's response. The mean is defined
  as the intensity level at which p(right)=0.5 in the definition of the Normal CDF.
  This is calculated online by taking the mean of the marginalized posterior over the threshold parameter.
- `mean_hdi`: The highest density interval (HDI) of the mean parameters, in pixels, for the given speed condition,
  after the current trial's response. This is calculated online using the marginalized
  posterior over the `mean` parameter, with `hdi` from `FLE_utils`.
- `sd_est`: The estimated standard deviation **parameter** of the participant's psychometric response function,
  in pixels, for the given speed condition, after the current trial's response. The standard deviation is defined
  in the Normal CDF function.
  This is calculated online by taking the mean of the marginalized posterior over the standard deviation parameter.
- `sd_hdi`: The highest density interval (HDI) of the standard deviation parameter, in pixels, for the given speed condition,
  after the current trial's response. This is calculated online by using the marginalized
  posterior over the `sd` parameter, with `hdi` from `FLE_utils`.
- `halfpoint_est`: Equal to `mean_est`. Legacy.
- `halfpoint_hdi`: Equal to `mean_hdi`. Legacy.
- `delay_est_ms`: The estimated neural delay of the participant, in milliseconds. This is calculated online by
  dividing the estimated halfpoint by the speed of the mover.
- `delay_hdi_ms`: The highest density interval (HDI) of the delay, in milliseconds, for the given speed condition,
  after the current trial's response. This is calculated online by reparameterizing the `mean` domain to `delay` by
  dividing by the speed of the mover, and using `hdi` from `FLE_utils`.

## `training.csv`

A table with a row for each training trial, with the following columns:

- `trial_num`: The training trial number.
- `success`: The success of the trial, as a boolean.
- `type`: The type of the trial, either `moving` or `stationary`.

## `attention_checks.csv`

A table with a row for each attention check, with the following columns:

- `check_num`: The attention check number.
- `trial_num`: The trial number that came right after the attention check.
- `success`: The success of the attention check, as a boolean.

## `extended_data/`

This folder contains a file for each speed condition named `{speed}.nc`. These files contain the full posterior and
prior distributions of the parameters of the psychometric response function, saved as a `netCDF4.Dataset` object. They
can be loaded using `xarray.load_dataset`. The dataset has a dictionary-like interface with the following keys:

- `prior` - an `xarray.DataArray` with the prior distribution of the parameters.
- `posterior` - an `xarray.DataArray` with the posterior distribution of the parameters.

## `{uid}.edf`
Eye-tracking data, if eye tracking was enabled. The file is in the EDF format, and can be converted to plaintext `.asc`
using the `edf2asc` utility from the [EyeLink Developers Kit](https://www.sr-research.com/support/thread-13.html)
(requires free forum registration). The `.asc` file is generated and parsed automatically by running 
`DataAnalyzer.prepare_et_data()`.

The `.asc` file contains rows of data, with most rows being eye location samples, and some being messages 
generated by the EyeLink or sent from the experiment program. We use pairs of these messages to delimit trials,
with trial start and end containing the row:

`MSG  {timestamp} TRIALID {N} {BEGIN|END}`

with the same `N`.

**Important note**: `N` is equal to the trial number and corresponds to `thisN` field in `results.csv` of the same trial. 

For a detailed description of the `.asc` format, see 
[this forum post](https://www.sr-research.com/support/thread-7675.html)
