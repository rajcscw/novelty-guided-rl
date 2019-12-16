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


class ESTrainer:
    def __init__(self, task_name, config, exp_tracker, device_list):
        self.task_name = task_name
        self.config = config
        self.exp_tracker = exp_tracker
        self.rl_weight = config["novelty"]["initial_rl_weight"]
        self.adaptive = config["novelty"]["adaptive"]
        self.max_iter = config["other"]["max_iter"]
        self.n_workers = config["other"]["n_workers"]
        self.log_every = config["other"]["log_every"]
        self.device_list = device_list
        self.n_workers = config["other"]["n_workers"]
        self.n_samples = config["other"]["n_samples"]
        self.strategy_name = self._get_strategy_name()

    def _create_novelty_detector(self):
        if self.config["novelty"]["technique"] == "KNNnovelty":
            return KNNModel.from_dict(self.config["KNNnovelty"])
        elif self.config["novelty"]["technique"] == "AEnovelty":
            config = self.config["AEnovelty"].copy()
            config.update({"n_input": self.config["behavior"]["traj_length"] * 2})
            config.update({"device": self.device_list[0]})
            return AEModel.from_dict(config)

    def _get_strategy_name(self):
        return self.config["novelty"]["technique"]

    @staticmethod
    def _infer_policy_parameters(env_name):
        env = GymEnvironment(env_name, 100) # create a dummy env
        state_dim, action_dim, policy_type = env.state_dim, env.action_dim, env.policy_type
        return state_dim, action_dim, policy_type

    def _get_all_components(self):
        state_dim, action_dim, policy_type = ESTrainer._infer_policy_parameters(self.config["environment"]["name"])

        # network
        net = PolicyNet(n_input=state_dim,
                        n_hidden=self.config["model"]["hidden_size"],
                        n_output=action_dim,
                        policy_type=policy_type)

        # novelty detector
        novelty_detector = self._create_novelty_detector()

        # set up the model
        model = PyTorchModel(net=net)

        # the objective function
        objective = EpisodicReturnPolicy(model=model,
                                         config=self.config)

        # estimator
        estimator = ES(parallel_workers=self.n_workers,
                       sigma=float(self.config["ES"]["sigma"]),
                       k=self.n_samples,
                       loss_function=objective,
                       device_list=self.device_list,
                       model=model,
                       rl_weight=self.rl_weight,
                       novelty_detector=novelty_detector)

        # novelty adaptor
        nov_adaptor = NoveltyAdaptor(rl_weight=self.rl_weight,
                                     rl_weight_delta=self.config["novelty"]["rl_weight_delta"],
                                     t_max=self.config["novelty"]["t_max"])

        return model, estimator, objective, novelty_detector, nov_adaptor

    def run(self):

        # get all components needed for training
        model, estimator, objective, novelty_detector, nov_adaptor = self._get_all_components()

        # the main loop
        episodic_total_reward = []
        reward_pressure = []
        running_total_reward = 0
        running_reward_pressure = 0.0
        for i in tqdm(range(self.max_iter)):
            total_reward = self.run_epoch(model, estimator, objective, novelty_detector, nov_adaptor)
            running_total_reward += total_reward
            running_reward_pressure += nov_adaptor.current_rl_weight

            if (i+1) % self.log_every == 0:
                running_total_reward = running_total_reward / self.log_every
                running_reward_pressure = running_reward_pressure / self.log_every

                self.exp_tracker.log(f"Processed Iteration {i+1},episodic return {running_total_reward}, reward_pressure: {running_reward_pressure}")
                running_total_reward = 0
                running_reward_pressure = 0

            # book keeping stuff
            episodic_total_reward.append(total_reward)
            reward_pressure.append(nov_adaptor.current_rl_weight)

        # close the pool now
        estimator.p.close()

        # save the final results
        self.exp_tracker.save_results({
            "episodic_total_rewards": rolling_mean(episodic_total_reward, self.log_every).tolist(),
            "reward_pressure": rolling_mean(reward_pressure, self.log_every).tolist()
        })

        # Log all the config parameters
        self.exp_tracker.save_config(self.config)

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