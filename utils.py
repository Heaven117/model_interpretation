import sys
import json
import logging
from pathlib import Path
from datetime import datetime as dt

import torch


def save_json(json_obj, json_path, append_if_exists=False,
              overwrite_if_exists=False, unique_fn_if_exists=True):
    """Saves a json file

    Arguments:
        json_obj: json, json object
        json_path: Path, path including the file name where the json object
            should be saved to
        append_if_exists: bool, append to the existing json file with the same
            name if it exists (keep the json structure intact)
        overwrite_if_exists: bool, xor with append, overwrites any existing
            target file
        unique_fn_if_exsists: bool, appends the current date and time to the
            file name if the target file exists already.
    """
    if isinstance(json_path, str):
        json_path = Path(json_path)

    if overwrite_if_exists:
        append_if_exists = False
        unique_fn_if_exists = False

    if unique_fn_if_exists:
        overwrite_if_exists = False
        append_if_exists = False
        if json_path.exists():
            time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
            json_path = json_path.parents[0] / f'{str(json_path.stem)}_{time}'\
                                               f'{str(json_path.suffix)}'

    if overwrite_if_exists:
        append_if_exists = False
        with open(json_path, 'w+') as fout:
            json.dump(json_obj, fout, indent=2)
        return

    if append_if_exists:
        if json_path.exists():
            with open(json_path, 'r') as fin:
                read_file = json.load(fin)
            read_file.update(json_obj)
            with open(json_path, 'w+') as fout:
                json.dump(read_file, fout, indent=2)
            return

    with open(json_path, 'w+') as fout:
        json.dump(json_obj, fout, indent=2)


def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step <= last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def init_logging(filename=None):
    """Initialises log/stdout output

    Arguments:
        filename: str, a filename can be set to output the log information to
            a file instead of stdout"""
    log_lvl = logging.INFO
    log_format = '%(asctime)s: %(message)s'
    if filename:
        logging.basicConfig(handlers=[logging.FileHandler(filename),
                                      logging.StreamHandler(sys.stdout)],
                            level=log_lvl,
                            format=log_format)
    else:
        logging.basicConfig(stream=sys.stdout, level=log_lvl,
                            format=log_format)

"""Returns a default config file"""
# def get_default_config():
#     model_config = {
#         'mode_name':'svm',
#         'dataset': 'FICO',
#         'device' : torch.device('cpu'),
#         'seed': 42,
#         'epoch' : 100,
#         'batch_size' : 5,
#         'lr' : 0.001,
#         'c' : 0.01,
#         'dataFile':'data/FICO_final_data.csv'
#     }
#     save_path='data/svm/'+f"svm_{model_config['dataset']}_{model_config['epoch']}.pth"
#     model_config['save_path'] = save_path

#     IF_config ={
#         'out_path': 'data/influence',
#         'recursion_depth': 10,
        
#     }

#     return model_config,IF_config
model_config = {
    'mode_name':'svm',
    'dataset': 'FICO',
    'device' : torch.device('cpu'),
    'seed': 42,
    'epoch' : 100,
    'batch_size' : 5,
    'lr' : 0.001,
    'c' : 0.01,
    'dataFile':'data/FICO_final_data.csv'
}
save_path='data/svm/'+f"svm_{model_config['dataset']}_{model_config['epoch']}.pth"

IF_config ={
    'out_path': 'data/influence',
    'recursion_depth': 10,
}

ft_names = ["External Risk Estimate", 
            "Months Since Oldest Trade Open",
            "Months Since Last Trade Open",
            "Average Months in File",
            "Satisfactory Trades",
            "Trades 60+ Ever",
            "Trades 90+ Ever",
            "% Trades Never Delq.",
            "Months Since Last Delq.",
            "Max Delq. Last 12M",
            "Max Delq. Ever",
            "Total Trades",
            "Trades Open Last 12M",
            "% Installment Trades",
            "Months Since Most Recent Inq",
            "Inq Last 6 Months",
            "Inq Last 6 Months exl. 7 days",
            "Revolving Burden",
            "Installment Burden",
            "Revolving Trades w/ Balance:",
            "Installment Trades w/ Balance",
            "Bank Trades w/ High Utilization Ratio",
            "% Trades w/ Balance"]
