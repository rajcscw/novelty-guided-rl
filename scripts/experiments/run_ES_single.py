from novelty_guided_package.core_components.es_trainer import ESTrainer
from novelty_guided_package.experiment_handling.tracker import ExperimentTracker
from novelty_guided_package.core_components.utility import init_multiproc
import sys
import json
import torch

def run(output_folder, task_config, run_id, use_gpu):
    device_count = torch.cuda.device_count()
    device_list = list(map(lambda x: "cuda:{}".format(x), range(device_count))) \
        if torch.cuda.is_available() and use_gpu else ["cpu"]

    task_config = json.loads(task_config)
    run_id = int(run_id)
    
    exp_tracker = ExperimentTracker(output_folder, exp_name=f"{run_id+1}")
    trainer = ESTrainer(task_config, exp_tracker, device_list)
    trainer.run()


if __name__ == "__main__":
    init_multiproc()
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])