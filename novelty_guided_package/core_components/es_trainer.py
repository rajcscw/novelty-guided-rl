import numpy as np
import gym
from tqdm import tqdm
import torch
from torch import nn
from novelty_guided_package.core_components.models import PyTorchModel
from novelty_guided_package.core_components.estimators import ES
from novelty_guided_package.core_components.loss import EpisodicReturnPolicy
from novelty_guided_package.core_components.SGDOptimizersES import RMSProp as RMSPropES
from novelty_guided_package.seq2seq_components.Seq2SeqNovelty import NoveltyDetectionModule as AEModel
from novelty_guided_package.novelty_components.KNNNovelty import NoveltyDetectionModule as KNNModel
from novelty_guided_package.novelty_components.adaptor import NoveltyAdaptor
from novelty_guided_package.core_components.utility import rolling_mean
from novelty_guided_package.core_components.networks import PolicyNet


class ESTrainer:
    def __init__(self, task_name, config, exp_tracker, device_list):
        self.task_name = task_name
        self.config = config
        self.exp_tracker = exp_tracker
        self.rl_weight = config["novelty"]["initial_rl_weight"]
        self.adaptive = config["novelty"]["adaptive"]
        self.device_list = device_list
        self.strategy_name = self._get_strategy_name()

        # it creates the environment here
        self.env, self.state_dim, self.action_dim = self._create_env(task_name, self.config)

    def _create_env(self, task_name, config):
        # create environments to get
        if self.config["environment"]["mini_grid"]:
            env = gym.make(self.config["environment"]["name"])
            action_dim = env.action_space.n
            state_dim = 7 * 7 * 3 # hard-coded for now
        else: # for other continuous control tasks
            env = gym.make(self.config["environment"]["name"])
            action_dim = env.action_space.shape[0]
            state_dim = env.observation_space.shape[0]
        return env, state_dim, action_dim

    def _create_novelty_detector(self):
        if self.config["novelty"]["technique"] == "SeqNovelty":
            return AEModel.from_dict(self.config["SeqNovelty"])
        elif self.config["novelty"]["technique"] == "KNNnovelty":
            return KNNModel.from_dict(self.config["KNNnovelty"])
        else:
            return None

    def _get_strategy_name(self):
        return self.config["novelty"]["technique"]

    def run(self):
        max_iter = int(self.config["log"]["iterations"])

        # policy types
        if self.config["policy"]["type"] == "gaussian":
            n_output = self.action_dim if self.config["policy"]["is_deterministic"] else self.action_dim * 2
        elif self.config["policy"]["type"] == "softmax":
            n_output = self.action_dim

        # network
        net = PolicyNet(n_input=self.state_dim,
                        n_hidden=self.config["model"]["hidden_size"],
                        n_output=n_output)

        # novelty detector
        novelty_detector = self._create_novelty_detector()

        # set up the model
        model = PyTorchModel(net=net)

        # the objective function
        objective = EpisodicReturnPolicy(model=model,
                                         env=self.env,
                                         config=self.config,
                                         state_dim=self.state_dim,
                                         action_dim=self.action_dim,
                                         is_deterministic=self.config["policy"]["is_deterministic"])

        # optimizer
        optimizer = RMSPropES(learning_rate=float(self.config["ES"]["a"]),
                              decay_rate=float(self.config["ES"]["momentum"]))

        # estimator
        estimator = ES(parallel_workers=int(self.config["ES"]["n_workers"]),
                       sigma=float(self.config["ES"]["sigma"]),
                       k=int(self.config["log"]["parallel_eval"]),
                       loss_function=objective,
                       device_list=self.device_list,
                       model=model,
                       rl_weight=self.rl_weight,
                       novelty_detector=novelty_detector)

        # novelty adaptor
        nov_adaptor = NoveltyAdaptor(rl_weight=self.rl_weight,
                                     rl_weight_delta=self.config["novelty"]["rl_weight_delta"],
                                     t_max=self.config["novelty"]["t_max"])

        # the main loop
        episodic_total_reward = []
        reward_pressure = []
        running_total_reward = 0
        running_reward_pressure = 0.0
        for i in tqdm(range(max_iter)):
            total_reward = self.run_epoch(model, estimator, optimizer, objective, novelty_detector, nov_adaptor)
            running_total_reward += total_reward
            running_reward_pressure += nov_adaptor.current_rl_weight

            if (i+1) % int(self.config["log"]["average_every"]) == 0:
                running_total_reward = running_total_reward / int(self.config["log"]["average_every"])
                running_reward_pressure = running_reward_pressure / int(self.config["log"]["average_every"])
                self.exp_tracker.log(f"Processed Iteration {i+1},episodic return {running_total_reward}, lr: {optimizer.get_learning_rate()}, reward_pressure: {running_reward_pressure}")
                running_total_reward = 0
                running_reward_pressure = 0

            # book keeping stuff
            episodic_total_reward.append(total_reward)
            reward_pressure.append(nov_adaptor.current_rl_weight)

            # step optimizer
            optimizer.step_t()

        # close the pool now
        estimator.p.close()

        # save the final results
        self.exp_tracker.save_results({
            "episodic_total_rewards": rolling_mean(episodic_total_reward, 20).tolist(),
            "reward_pressure": rolling_mean(reward_pressure, 20).tolist()
        })

        # Log all the config parameters
        self.exp_tracker.save_config(self.config)

    def run_epoch(self, model, estimator, optimizer, objective, novelty_detector, novelty_adaptor):
        # get the current parameter name and value
        current_layer_name, current_layer_value = model.sample_layer()

        # estimator
        gradients, behaviors = estimator.estimate_gradient(current_layer_name, current_layer_value)

        # step using default optimizer
        model.step_layer(current_layer_name, -gradients)

        # seed for evaluation (same type of generation as in estimators)
        seed = np.random.randint(0, int(self.config["log"]["parallel_eval"]/2))

        # evaluate the learning
        objective.parameter_name = current_layer_name
        total_reward, current_behavior, _ = objective(None, self.device_list[0], seed, True)

        # novelty model - a mini-batch step
        if novelty_detector is not None:

            # can we just add this behavior for the learning instead of one from all the perturbations
            estimator.novelty_detector.fit_model([current_behavior])

            novelty_detector.step()

        # here, we adapt the reward/novelty weights based on stagnation
        if self.adaptive:
            novelty_adaptor.adapt(total_reward)

        # set new parameters
        estimator.rl_weight = novelty_adaptor.current_rl_weight

        # step optimizer
        optimizer.step_t()

        # return the total reward
        return total_reward