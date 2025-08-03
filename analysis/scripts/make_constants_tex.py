import argparse
import json
import os
import math


def format_pval(pval, max_d=2, min_d=20):
    for d in range(min_d, max_d - 1, -1):
        if pval < (10 ** (-d)):
            return "p < 10^{{-{}}}".format(d)
    return "p = {}".format(round(pval, max_d))


def format_value_with_error(value: float, error: float) -> str:
    if error == 0:
        return f"{value:.6g} ± 0"  # Arbitrary precision if no error

    # Find order of magnitude of error
    oom_err = math.floor(math.log10(abs(error)))
    first_sig_digit = round(error / (10**oom_err))  # First significant digit of error
    sig_err = first_sig_digit * (10**oom_err)  # Rounded error to 1 sig fig

    # Round value to the same decimal place as the rounded error
    decimal_places = -oom_err if oom_err < 0 else 0
    rounded_value = round(value, decimal_places)

    # Formatting based on decimal places
    fmt = f"{{:.{decimal_places}f}}" if decimal_places > 0 else "{:.0f}"

    return fr"{fmt.format(rounded_value)}\pm{fmt.format(sig_err)}"


def load_lag_speed(path):
    data = json.load(open(path))
    lag_speed = {
        'figTwoSlope': "$" + format_value_with_error(data['delay (ms)'], error=data['delay std (ms)']) + '$ ms',
        'lagSpeedMaxP': "$" + format_pval(data['max p-value']) + "$",
    }
    return lag_speed


def load_bfc_noise(path):
    data = json.load(open(path))
    bfc_noise = {
        'figThreeSlope': format_value_with_error(data['slope'], error=data['slope_std']),
        'bfcNoiseP': format_pval(data['pvalue']),
    }
    for key, val in bfc_noise.items():
        bfc_noise[key] = "$" + val + "$"
    return bfc_noise


def load_sig_sig(path):
    data = json.load(open(path))
    rho_dec_places = 2
    sig_sig = {
        'figSigSlope': format_value_with_error(data['slope'], error=data['slope_std']),
        'sigSigRho': f'{data["rho"]:.{rho_dec_places}f}',
        'sigSigCI': rf'\left[{data["rho_CI"][0]:.{rho_dec_places}f}, {data["rho_CI"][1]:.{rho_dec_places}f}\right]',
        'sigSigP': format_pval(data["p_value"]),
    }
    for key, val in sig_sig.items():
        sig_sig[key] = "$" + val + "$"
    return sig_sig


def load_segmentation(path):
    data = json.load(open(path))
    segmentation = {
        'rampTime': f'{data['ramp_time']} ms',
        'minLength': f'{data['min_length']} ms',
        'saccadeSpeedThresh': f'{data['saccade_speed_thresh']} pixels/ms',
        'inflation': f'{data['inflation']} ms',
        'smoothingWindow': f'{data['window']} ms',
        'slopeSpeedFactor': f'{data['slope_thresh_factor']}',
    }
    return segmentation


def load_exp_params(path):
    data = json.load(open(path))
    exp_params = {
        'nTrialsPerSpeed': f'{data['n_trials']}',
        'breakEveryN': f'{data['break_every']}',
        'attnCheckEveryN': f'{data['attention_check_every']}',
    }
    flasher_presentation_time = (data['flash_duration_ms'] -
                                 data['flash_duration_ms'] % (data['frame_duration_sec'] * 1000))
    exp_params['flasherTime'] = f'{int(flasher_presentation_time)} ms'
    return exp_params


def load_exclusion_params(path):
    data = json.load(open(path))
    exclusion_params = {
        'attnCheckExclude': f'{data['attention check threshold']}',
    }
    return exclusion_params


def count_participants(data_path, exclusions_path):
    n_total = len(os.listdir(data_path))
    exclusions = json.load(open(exclusions_path))
    n_excluded = len(set([str(p) for reason, participants in exclusions.items() for p in participants]))
    return n_total, n_total - n_excluded


def count_et_participants(data_path, exclusions_path):
    n_total, n_after_exclusions = count_participants(data_path, exclusions_path)
    return {
        'nParticipantsET': f'{n_total}',
        'nParticipantsETAfterExclusions': f'{n_after_exclusions}',
    }


def count_vanilla_participants(data_path, exclusions_path):
    n_total, n_after_exclusions = count_participants(data_path, exclusions_path)
    return {
        'nParticipantsVanilla': f'{n_total}',
        'nParticipantsVanillaAfterExclusions': f'{n_after_exclusions}',
    }


def generate_tex_macro(name, content, comment=None):
    result = f"\\newcommand{{\\{name}}}{{{content}\\xspace}}"
    if comment:
        result = f"{result} % {comment}"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None)
    parser.add_argument('--exp-dir', default=None)
    args = parser.parse_args()
    root = args.path
    exp_dir = args.exp_dir

    exp_params = load_exp_params(os.path.join(exp_dir, r'FLE_exp4_full_vanilla-experiment-2024\raw\2024-05-12_12h05m04s_31112707/params.json'))
    cohort_params_et = count_et_participants(os.path.join(exp_dir, 'FLE_exp3_eyetracker-experiment-2024/raw'),
                                             os.path.join(root, 'exclusions_et.json'))
    cohort_params_vanilla = count_vanilla_participants(os.path.join(exp_dir, 'FLE_exp4_full_vanilla-experiment-2024/raw'),
                                                       os.path.join(root, 'exclusions_vanilla.json'))

    lag_speed = load_lag_speed(os.path.join(root, 'lag_vs_speed.json'))
    bfc_noise = load_bfc_noise(os.path.join(root, 'bfc_noise.json'))
    sig_sig = load_sig_sig(os.path.join(root, 'sig-sig_r.json'))

    segmentation_params = load_segmentation(os.path.join(exp_dir, 'FLE_exp3_eyetracker-experiment-2024/processed/epoch_segmentation_params.json'))

    exclusion_params = load_exclusion_params(os.path.join(root, 'exclusion_params.json'))

    all_constants = {}
    all_constants.update(cohort_params_vanilla)
    all_constants.update(cohort_params_et)
    all_constants.update(exp_params)
    all_constants.update(lag_speed)
    all_constants.update(bfc_noise)
    all_constants.update(sig_sig)
    all_constants.update(segmentation_params)
    all_constants.update(exclusion_params)
    lines = []
    for name, content in all_constants.items():
        lines.append(generate_tex_macro(name, content))

    output_file = "constants.tex"
    try:
        with open(output_file, "w") as f:
            for line in lines:
                f.write(line + "\n")
            print(f"Successfully wrote {len(lines)} constants to {output_file}")
    except Exception as e:
        print(f"Failed to write to {output_file}: {e}")
        return 1
    return 0


if __name__ == '__main__':
    main()
