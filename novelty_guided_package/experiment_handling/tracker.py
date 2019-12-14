import logging
import os
import json


class ExperimentTracker:
    def __init__(self, base_folder, exp_name="novelty"):
        self.exp_name = exp_name
        self.base_folder = self._set_up_paths(base_folder)
        self.logger = self._set_up_logger(self.base_folder, exp_name)

    def _set_up_paths(self, base_folder):
        folder_name = os.path.join(base_folder, self.exp_name)
        os.makedirs(folder_name)
        return folder_name

    @staticmethod
    def _set_up_logger(base_folder, name):
        logger = logging.Logger(name)
        logger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.FileHandler(os.path.join(base_folder, "console.log")))
        return logger

    def log(self, message):
        self.logger.log(level=logging.INFO, msg=message)

    def save_model(self, model):
        pass

    def save_config(self, config):
        json.dump(config, open(os.path.join(self.base_folder, "config.json"), "w"))

    def save_results(self, results):
        path_to_results = os.path.join(self.base_folder, "metrics.json")
        prev_results = {} if not os.path.exists(path_to_results) else json.load(open(path_to_results))
        prev_results.update(results)
        json.dump(prev_results, open(path_to_results, "w"))

