from tqdm import tqdm
from novelty_guided_package.core_components.models import PyTorchModel
from novelty_guided_package.core_components.estimators import ES
from novelty_guided_package.core_components.objective import EpisodicReturnPolicy
from novelty_guided_package.novelty_components.KNNNovelty import NearestNeighborDetection as KNNModel
from novelty_guided_package.novelty_components.AENovelty import AutoEncoderBasedDetection as AEModel
from novelty_guided_package.novelty_components.adaptor import NoveltyAdaptor
from novelty_guided_package.core_components.networks import PolicyNet
from novelty_guided_package.environments.gym_wrappers import GymEnvironment
from novelty_guided_package.core_components.utility import rolling_mean
from novelty_guided_package.environments.gym_wrappers import PolicyType


class ESTrainer:
    def __init__(self,
                 config,
                 exp_tracker,
                 device_list):
        self.config = config
        self.device_list = device_list
        self.exp_tracker = exp_tracker
        self._extract_config_params()

    def _extract_config_params(self):
        # general es config and run time parameters
        self.task_name = self.config["environment"]["name"]
        self.max_episode_steps = self.config["environment"]["max_episode_steps"]
        self.max_iter = self.config["run_time"]["max_iter"]
        self.n_workers = self.config["run_time"]["n_workers"]
        self.log_every = self.config["run_time"]["log_every"]
        self.n_workers = self.config["run_time"]["n_workers"]
        self.n_samples = self.config["run_time"]["n_samples"]
        self.sigma = float(self.config["ES"]["sigma"])
        self.lr = float(self.config["ES"]["lr"])
        self.stochastic = self.config["environment"]["stochastic"]

        # adaptive
        self.adaptive = self.config["method"]["adaptive"]["adapt"]
        self.rl_weight = self.config["method"]["adaptive"]["initial_rl_weight"]
        self.rl_weight_delta = self.config["method"]["adaptive"].get("rl_weight_delta", None)
        self.t_max = self.config["method"]["adaptive"].get("t_max", None)

        # novelty
        self.novelty_detector_config = self.config["method"].get("novelty_detector", None)
        self.novelty_detector_name = self.novelty_detector_config["name"] \
            if self.novelty_detector_config is not None else None
        self.behavior_traj_length = self.config["method"].get("behavior_traj_length", None)
        self.behavior_dim = self.config["method"].get("behavior_dim", None)

        # policy network
        self.policy_hidden_size = self.config["model"]["hidden_size"]

    def _create_novelty_detector(self):
        if self.novelty_detector_name == "knn":
            config = self.novelty_detector_config.copy()
            del config["name"]
            return KNNModel.from_dict(config)
        elif self.novelty_detector_name == "ae":
            config = self.novelty_detector_config.copy()
            del config["name"]
            config.update({"n_input": self.behavior_traj_length *
                                      self.behavior_dim})
            config.update({"device": self.device_list[0]})  # TBD
            return AEModel.from_dict(config)
        else:
            return None

    @staticmethod
    def _infer_policy_parameters(env_name, stochastic):
        env = GymEnvironment(env_name, 100)  # create a dummy env to get the spaces
        state_dim, action_dim, policy_type = env.state_dim, env.action_dim, env.policy_type
        action_dim = action_dim * 2 if (env.policy_type == PolicyType.GAUSSIAN and stochastic) else action_dim
        return state_dim, action_dim, policy_type

    def _get_all_components(self):
        state_dim, action_dim, policy_type = ESTrainer._infer_policy_parameters(self.task_name, self.stochastic)

        # network
        net = PolicyNet(n_input=state_dim,
                        n_hidden=self.policy_hidden_size,
                        n_output=action_dim,
                        policy_type=policy_type)

        # novelty detector
        novelty_detector = self._create_novelty_detector()

        # set up the model
        model = PyTorchModel(net=net, lr=self.lr)

        # the objective function
        objective = EpisodicReturnPolicy(model=model,
                                         task_name=self.task_name,
                                         max_episode_steps=self.max_episode_steps,
                                         behavior_traj_length=self.behavior_traj_length)

        # estimator
        estimator = ES(parallel_workers=self.n_workers,
                       sigma=self.sigma,
                       k=self.n_samples,
                       loss_function=objective,
                       device_list=self.device_list,
                       model=model,
                       rl_weight=self.rl_weight,
                       novelty_detector=novelty_detector)

        # novelty adaptor
        if self.adaptive:
            nov_adaptor = NoveltyAdaptor(rl_weight=self.rl_weight,
                                         rl_weight_delta=self.rl_weight_delta,
                                         t_max=self.t_max)
        else:
            nov_adaptor = None

        return model, estimator, objective, novelty_detector, nov_adaptor

    def run(self):

        # Log all the config parameters
        self.exp_tracker.save_config(self.config)
        self.exp_tracker.save_results({})

        # get all components needed for training
        model, estimator, objective, novelty_detector, nov_adaptor = self._get_all_components()

        # the main loop
        episodic_total_reward = []
        reward_pressure = []
        running_total_reward = 0
        running_reward_pressure = 0.0
        for i in range(self.max_iter):
            total_reward = self.run_epoch(model, estimator, objective, novelty_detector, nov_adaptor)
            running_total_reward += total_reward
            running_reward_pressure += estimator.rl_weight

            if (i+1) % self.log_every == 0:
                running_total_reward = running_total_reward / self.log_every
                running_reward_pressure = running_reward_pressure / self.log_every

                self.exp_tracker.log(f"Processed Iteration {i+1},episodic return {running_total_reward}, "
                                     f"reward_pressure: {running_reward_pressure}")
                running_total_reward = 0
                running_reward_pressure = 0

            # book keeping stuff
            episodic_total_reward.append(total_reward)
            reward_pressure.append(estimator.rl_weight)

        # close the pool now
        estimator.p.close()

        # save the final results
        self.exp_tracker.save_results({
            "episodic_total_rewards": rolling_mean(episodic_total_reward, self.log_every).tolist(),
            "reward_pressure": rolling_mean(reward_pressure, self.log_every).tolist()
        })

    def run_epoch(self, model, estimator, objective, novelty_detector, novelty_adaptor):
        # get the current parameter name and value
        current_layer_name, current_layer_value = model.sample_layer()

        # estimator
        gradients, behaviors = estimator.estimate_gradient(current_layer_name, current_layer_value)

        # step using default optimizer
        model.step_layer(current_layer_name, -gradients)

        # evaluate the learning
        objective.parameter_name = current_layer_name
        total_reward, current_behavior, _ = objective(None, self.device_list[0])

        # novelty model - a mini-batch step
        if novelty_detector is not None:

            # can we just add this behavior for the learning instead of one from all the perturbations
            estimator.novelty_detector.save_behaviors([current_behavior])

            novelty_detector.step()

        # here, we adapt the reward/novelty weights based on stagnation
        if self.adaptive:
            novelty_adaptor.adapt(total_reward)

            # set new parameters
            estimator.rl_weight = novelty_adaptor.current_rl_weight

        # return the total reward
        return total_reward