import os
import json
import torch
from novelty_guided_package.core_components.utility import init_multiproc
from novelty_guided_package.core_components.es_trainer import ESTrainer
from novelty_guided_package.experiment_handling.tracker import ExperimentTracker
from novelty_guided_package.core_components.grid_search import split_config
from datetime import datetime
import argparse
import multiprocessing as mp
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import glob


def main(logdir, config_path, n_samples, n_workers, n_runs, max_iter, average_every, use_gpu):
    device_count = torch.cuda.device_count()
    device_list = list(map(lambda x: "cuda:{}".format(x), range(device_count))) \
        if torch.cuda.is_available() and use_gpu else ["cpu"]

    # time
    start_time = str(datetime.now())

    # get the config here
    gs_config = json.load(open(config_path))

    # split the configs
    task_configs = split_config(gs_config)

    # repeat for n_runs
    task_configs = np.repeat(task_configs, n_runs).tolist()
    shuffle(task_configs)

    print(f"Running the experiment for {len(task_configs)} configs")

    for run_id, task_config in enumerate(tqdm(task_configs)):

        task_name = task_config["environment"]["name"]
        method_type = task_config["method"]["name"]
        task_config["run_time"] = {}
        task_config["run_time"]["n_workers"] = n_workers
        task_config["run_time"]["log_every"] = average_every
        task_config["run_time"]["max_iter"] = max_iter
        task_config["run_time"]["n_samples"] = n_samples
        output_folder = os.path.join(logdir, start_time, method_type, task_name)

        # invoke the single script with output folder, experiment_id, config, use_gpu
        task_config = json.dumps(json.dumps(task_config))
        output_folder = output_folder.replace(" ", "_")
        os.system(f"python run_ES_single.py {output_folder} {task_config} {run_id+1} {use_gpu}")

if __name__ == "__main__":
    init_multiproc()

    # read the argument
    parser = argparse.ArgumentParser("Run ES experiments...")
    parser.add_argument("--config", type=str, help="path to the config json")
    parser.add_argument("--logdir", type=str, help="directory to log the results")
    parser.add_argument("--runs", type=int, help="number of times to repeat the experiment")
    parser.add_argument("--n_sample", type=int, help="sample size of ES")
    parser.add_argument("--max_iter", type=int, help="maximum number of iterations")
    parser.add_argument("--average_every", type=int, default=50, help="running average for tracking the stats")
    parser.add_argument("--n_workers", type=int, default=mp.cpu_count(), help="number of CPU workers to allocate")
    parser.add_argument("--use_gpu", default=False, help="to use GPU or not")
    args = parser.parse_args()
    main(args.logdir, args.config, args.n_sample, args.n_workers, args.runs, args.max_iter, args.average_every, args.use_gpu)