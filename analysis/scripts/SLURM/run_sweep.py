import itertools
from datetime import datetime
import time
import argparse
import os
import sys
import logging

import yaml

import numpy as np
import pandas as pd
from dask_jobqueue import SLURMCluster
from dask import delayed

from analysis.gaussian_process import llk as calc_llk
from analysis.bayesian_inference_et import VelVelControlKernel, VelVelControlMeanProcess
from util.et_utils import epoch_v


def setup_logging(name):
    """
    Configure logging system for the script.

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Create handlers for stdout and stderr
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)

    # Set levels for handlers
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setLevel(logging.WARNING)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger


def load_conf(config_path):
    """
    Load the configuration file.

    :param config_path: Path to the configuration file.
    """

    logging.info(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_venv_activation_path():
    """
    Get the path to the virtual environment activation script.

    :return: Path to the activation script.
    """

    venv_root = sys.prefix
    if os.name == 'posix':
        activation_script = os.path.join(venv_root, 'bin', 'activate')
    elif os.name == 'nt':
        activation_script = os.path.join(venv_root, 'Scripts', 'activate.bat')
    else:
        logging.error("Unsupported OS")
        raise OSError("Unsupported OS")
    return activation_script


def load_data(data_dir):
    """
    Load data from the specified directory.

    :param data_dir: Directory containing the data files.
    :return: List of DataFrames for each epoch.
    """

    logging.info(f"Loading data from {data_dir}")
    epochs = []
    for epoch_file in sorted(os.listdir(data_dir)):
        epoch = pd.read_csv(os.path.join(data_dir, epoch_file))
        epochs.append(epoch)
    return epochs


# Define a function to submit to Dask, which takes a single parameter set
def llk_from_params(epochs, K, b, sig_r, sig_y):
    """
    Calculate the log-likelihood for a given set of parameters.

    :param epochs: List of DataFrames for each epoch.
    :param K: Control gain parameter.
    :param b: Control bias parameter.
    :param sig_r: Perceptual noise parameter.
    :param sig_y: Eye tracking noise parameter.
    :return: List of log-likelihood values for each epoch.
    """

    kern = VelVelControlKernel(K=K, sig_r=sig_r, sig_y=sig_y, b=b, large_t=True)
    mean_proc = VelVelControlMeanProcess(K=K, b=b, large_t=True)
    llks = []
    for epoch in epochs:
        t = epoch['trial_time'].values
        v = epoch_v(epoch, window=1, assume_equidistant=True) / 1000
        # fix nan values at edges
        v.iloc[0] = v.iloc[1]
        v.iloc[-1] = v.iloc[-2]
        t = t[::resample]
        v = v[::resample]

        mean = mean_proc(t, v=v_t / 1000)
        llks.append(calc_llk(t, v, mean, kern))
    return llks


def run_task_chunk(epochs, param_chunk):
    """
    Run a chunk of tasks with the given parameters.

    :param epochs: List of DataFrames for each epoch.
    :param param_chunk: List of parameter sets to evaluate.
    :return: DataFrame with the results of the log-likelihood calculations.
    """

    all_results = []
    for i, params in enumerate(param_chunk):
        logging.info(f"Running task {i} with params {params}")
        llks = llk_from_params(epochs, **params)
        for jj, val in enumerate(llks):
            res = params.copy()
            res['epoch'] = jj
            res[RES_VAR_NAME] = val
            all_results.append(res)
    return pd.DataFrame(all_results).set_index(list(param_chunk[0].keys()) + ['epoch'])


if __name__ == '__main__':
    logger = setup_logging('run_sweep')
    RES_VAR_NAME = 'llk'

    logger.info("Starting...")
    parser = argparse.ArgumentParser(description='Run a parameter sweep for the ET model.')
    parser.add_argument('--root_oath', type=str, help='Path to the project directory.')
    parser.add_argument('--v_t', type=float, required=True, help='The target velocity for the model.')
    parser.add_argument('--config', default=None, type=str, help='Path to the config file.')
    parser.add_argument('--id', type=str, help='Participant ID.')
    args = parser.parse_args()

    v_t = args.v_t
    root_path = args.root_path
    config_path = args.config or 'config.yaml'
    config = load_conf(config_path)
    part_id = args.id

    # load epochs
    logger.info(f"Loading epochs for participant {part_id}, speed {int(v_t)}")
    data_dir = f'{root_path}/tracking_epochs/{part_id}/{int(v_t)}'
    epochs = load_data(data_dir)

    # Define parameter ranges
    coords = {}
    param_names = ['K', 'b', 'sig_r', 'sig_y']
    for name in param_names:
        min_range = config['grid'][name]['min']
        max_range = config['grid'][name]['max']
        n = config['grid'][name]['n']
        coords[name] = np.geomspace(min_range, max_range, n)
    param_combos = [dict(zip(param_names, values)) for values in itertools.product(*(coords[key] for key in param_names))]
    chunk_size = int(np.ceil(1000 / len(epochs)))
    param_chunks = [param_combos[i:i + chunk_size] for i in range(0, len(param_combos), chunk_size)]

    resample = config['resamples'][int(v_t)]

    logger.info('Starting Dask cluster...')
    cluster = SLURMCluster(
        job_script_prologue=[
            f'source {get_venv_activation_path()}',
        ],
    )

    num_jobs = min(config['num_jobs'], len(param_combos))
    logger.info(f"Scaling cluster to {num_jobs} jobs")
    sys.stdout.flush()
    logger.debug(cluster.job_script())
    cluster.scale(jobs=num_jobs)
    client = cluster.get_client()
    logger.info(f"Dashboard link: {client.dashboard_link}")
    sys.stdout.flush()
    # Submit tasks for each parameter combination
    tasks = [delayed(run_task_chunk)(epochs, chunk) for chunk in param_chunks]
    start = time.time()
    futures = client.compute(tasks)  # Submit all tasks to the cluster
    stop = time.time()
    logger.info(f"Time to compute: {stop - start}")
    results = client.gather(futures)  # Collect all results
    stop_2 = time.time()
    logger.info(f"Time to gather: {stop_2 - stop}")
    logger.info("Gathering complete")

    # Fill the DataArray with results
    result_pd = pd.concat(results)
    result_array = result_pd.to_xarray()[RES_VAR_NAME]

    # Add metadata to the DataArray
    result_array.attrs['v_t'] = v_t
    result_array.attrs['resample'] = resample
    result_array.attrs['part_id'] = part_id
    timestamp = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
    result_array.attrs['timestamp'] = timestamp

    # Save the final DataArray to disk as a NetCDF file
    res_dir = f'{root_path}/et_llk/{part_id}'
    os.makedirs(res_dir, exist_ok=True)
    filename = f"{timestamp}_{int(v_t)}.nc"
    out_path = os.path.join(res_dir, filename)
    result_array.to_netcdf(out_path)
    logger.info(f"Saved results to {out_path}")

    # Shut down Dask client after computations are done
    client.close()
    cluster.close()
    logger.info("Cluster and client shut down")
