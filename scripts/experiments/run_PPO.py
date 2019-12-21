import os
import json
import torch
from novelty_guided_package.core_components.utility import init_multiproc
from novelty_guided_package.core_components.es_trainer import ESTrainer
from novelty_guided_package.experiment_handling.tracker import ExperimentTracker
from datetime import datetime
import argparse
import multiprocessing as mp


def main(config_path, n_samples, n_workers, n_runs, max_iter, average_every, use_gpu):
    device_count = torch.cuda.device_count()
    device_list = list(map(lambda x: "cuda:{}".format(x), range(device_count))) if torch.cuda.is_available() and use_gpu else ["cpu"]

    # get the config
    task_config = json.load(open(config_path))

    task_name = task_config["environment"]["name"]

    # time
    start_time = str(datetime.now())

    # now, run the task
    output_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs", start_time, task_name)

    for run_id in range(int(n_runs)):
        exp_tracker = ExperimentTracker(output_folder, exp_name=f"run_{run_id+1}")
        trainer = ESTrainer(task_name, task_config, exp_tracker, device_list)
        trainer.run()


if __name__ == "__main__":
    init_multiproc()

    # read the arguments
    parser = argparse.ArgumentParser("Run PPO experiments...")
    parser.add_argument("--config", type=str, help="path to the config json")
    parser.add_argument("--runs", type=int, default=5, help="number of times to repeat the experiment")
    parser.add_argument("--max_iter", type=int, default=2000, help="maximum number of iterations")
    parser.add_argument("--average_every", type=int, default=50, help="running average for tracking the stats")
    args = parser.parse_args()
    main(args.config, args.n_sample, args.n_workers, args.runs, args.max_iter, args.average_every, args.use_gpu)