import torch
from torch import nn
from components.core_components.models import SamplingStrategy
from components.core_components.models import PyTorchModel
from components.core_components.estimators import ES
from components.core_components.loss import EpisodicReturnPolicy
from components.core_components.SGDOptimizersES import RMSProp as RMSPropES
from components.core_components.adaptive import adapt_novelty_pressure
import numpy as np
import pandas as pd


# set up the network architecture
class PolicyNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def run_experiment_gym_with_es(env, config, state_dim, action_dim, rl_weight, adaptive, folder_name, novelty_detector, behavior, device_list, behavior_file):
    """
    :param config: takes config parameters
    :param rl_weight: RL pressure
    :param state_space: state space
    :param action_space: action space
    :param adaptive: True or False indicating whether to adapt the RL/Novelty weight while learning
    :return: returns a list of episodic reward over time
    """

    # Other variables
    max_iter = int(config["log"]["iterations"])

    # sampling strategy
    config["strategy"] = SamplingStrategy.ALL

    # policy types
    if config["policy"]["type"] == "gaussian":
        n_output = action_dim if config["policy"]["is_deterministic"] else action_dim * 2
    elif config["policy"]["type"] == "softmax":
        n_output = action_dim

    # network
    net = PolicyNet(n_input=state_dim,
                    n_hidden=config["model"]["hidden_size"],
                    n_output=n_output)

    # set up the model
    model = PyTorchModel(net=net, strategy=config["strategy"])

    # Scale the inputs
    scaler = None

    # the objective function
    objective = EpisodicReturnPolicy(model=model,
                                     env=env,
                                     config=config,
                                     scaler=scaler,
                                     behavior_selection=behavior,
                                     state_dim=state_dim,
                                     action_dim=action_dim,
                                     is_deterministic=config["policy"]["is_deterministic"])

    # optimizer
    optimizer = RMSPropES(learning_rate=float(config["ES"]["a"]),
                          decay_rate=float(config["ES"]["momentum"]))

    # estimator
    estimator = ES(parallel_workers=int(config["ES"]["n_workers"]),
                   sigma=float(config["ES"]["sigma"]),
                   k=int(config["log"]["parallel_eval"]),
                   loss_function=objective,
                   device_list=device_list,
                   model=model,
                   rl_weight=rl_weight,
                   novelty_detector=novelty_detector)

    # the main loop
    episodic_total_reward = []
    episodic_behaviors = pd.DataFrame()
    running_total_reward = 0
    f_best = -np.inf
    t_best = 0
    reward_pressure = []
    running_reward_pressue = 0.0
    for i in range(max_iter):
        # get the current parameter name and value
        current_layer_name, current_layer_value = model.sample_layer()

        # estimator
        gradients, behaviors = estimator.estimate_gradient(current_layer_name, current_layer_value)

        if behavior_file is not None:
            for j, perturb_beh in enumerate(behaviors):
                behavior_file.create_dataset("epoch_{}_{}".format(i, j), data=perturb_beh.numpy() if type(perturb_beh) == torch.Tensor else perturb_beh)

        # optimizer
        updated_layer_value = optimizer.step(current_layer_name, current_layer_value, gradients)

        # update the parameter
        model.set_layer_value(current_layer_name, updated_layer_value)

        # seed for evaluation (same type of generation as in estimators)
        seed = np.random.randint(0, int(config["log"]["parallel_eval"]/2))

        # evaluate the learning
        objective.parameter_name = current_layer_name
        total_reward, current_behavior, _ = objective(updated_layer_value, device_list[0], seed, True)
        running_total_reward += total_reward

        # novelty model - a mini-batch step
        if novelty_detector is not None:

            # can we just add this behavior for the learning instead of one from all the perturbations
            estimator.novelty_detector.fit_model([current_behavior])

            novelty_detector.step()

        # here, we adapt the reward/novelty weights based on stagnation
        if adaptive:
            # get current parameters
            current_f = total_reward
            current_rl_weight = estimator.rl_weight

            # get adapted and new parameters
            new_rl_weight, f_best, t_best = adapt_novelty_pressure(current_f=current_f,
                                                                   f_best=f_best,
                                                                   t_best=t_best,
                                                                   current_rl_weight=current_rl_weight,
                                                                   rl_weight_delta=config["novelty"]["rl_weight_delta"],
                                                                   t_max=config["novelty"]["t_max"])

            # set new parameters
            estimator.rl_weight = new_rl_weight

            # running reward
            running_reward_pressue += new_rl_weight
        else:
            running_reward_pressue += rl_weight

        # log the reward pressure
        reward_pressure.append(estimator.rl_weight)

        print("\rProcessed iteration {} of {}".format(i+1, max_iter), end="")
        if (i+1) % int(config["log"]["average_every"]) == 0:
            running_total_reward = running_total_reward / int(config["log"]["average_every"])
            running_reward_pressue = running_reward_pressue / int(config["log"]["average_every"])
            print(",Evaluating at iteration: {}, episodic return {}, lr: {}, reward_pressure: {}".format(i+1, running_total_reward, optimizer.get_learning_rate(), running_reward_pressue))
            running_total_reward = 0
            running_reward_pressue = 0

        # book keeping stuff
        episodic_total_reward.append(total_reward)

        # step optimizer
        optimizer.step_t()

        # save the model
        if i == 0 or (i+1) % config["log"]["save_every"] == 0:
            torch.save(model.net.cpu(), folder_name+"/model_"+"{0:0=4d}".format(i if i == 0 else i + 1))

    # behavior_file
    if behavior_file is not None:
        behavior_file.close()

    # close the pool now
    estimator.p.close()

    # return the results
    return episodic_total_reward
