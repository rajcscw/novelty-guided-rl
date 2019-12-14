import numpy as np
import gym
gym.logger.set_level(40)
import torch
from torch.nn import Softmax
from novelty_guided_package.environments.gym_wrappers import GymEnvironment, PolicyType
from novelty_guided_package.core_components.utility import to_device, get_equi_spaced_points


class EpisodicReturnPolicy:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.eval = eval

        # dummy variables to hold current parameter name
        self.parameter_name = None
        self.add_noise = False

        # Set up the model
        self.model = model

    def get_best_action(self, env, state, device):

        # choose the action
        state = torch.from_numpy(state.flatten()).type(torch.float32).view(1,-1)

        # send it to the device
        state = to_device(state, device)

        network_output = self.model.net.forward(state)

        if env.policy_type == PolicyType.GAUSSIAN:
            # apply gaussian policy
            action = self.gaussian_policy(network_output)

            # apply the bounds
            a_min = env.action_space.low.reshape((-1,1))
            a_max = env.action_space.high.reshape((-1,1))
            action = np.clip(action, a_min, a_max)
        elif env.policy_type == PolicyType.SOFTMAX:
            # apply softmax
            action = self.softmax_policy(network_output)

        return action

    def softmax_policy(self, network_output):
        network_output = Softmax(dim=1)(network_output)
        action = torch.multinomial(input=network_output, num_samples=1).cpu().data.numpy()
        return action

    def gaussian_policy(self, network_output):

        # break them into mean (mu) and sigma of gaussian distribution
        network_output = network_output.cpu().data.numpy().flatten()
        mean, sd = network_output[:self.n_a], network_output[self.n_a:]

        if self.is_deterministic:
            action = mean.reshape((self.n_a, 1))
        else:
            sd = np.log(1 + np.exp(sd))
            action = np.random.normal(loc=mean.reshape(-1,1), scale=sd.reshape(-1,1), size=(self.n_a,1))

        return action

    def __call__(self,
                 parameter_value,
                 device,
                 render=False,
                 save_loc=None):

        # let's create the env here
        env = GymEnvironment(self.config["environment"]["name"],
                             self.config["environment"]["max_episode_steps"],
                             render,
                             save_loc)

        # build a model with this parameter
        if parameter_value is not None:
            self.model.set_layer_value(self.parameter_name, parameter_value)

        # pass the model to the device
        self.model.net = to_device(self.model.net, device)

        # now we actually play an episode with the environment
        current_state = env.reset()

        # get the init beh
        init_beh = env.get_behavior(init_beh=None)

        episodic_return = 0
        stats = []
        trajectory = []
        steps_taken = 0
        while True:
            # Choose the best action
            current_action = self.get_best_action(env, current_state, device)

            # Perform the chosen action and get the reward and next state of the environment
            next_state, current_reward, done, info = env.step(current_action.flatten())
            stats.append(info)
            steps_taken += 1
            episodic_return += current_reward

            if done:
                break

            # Set the next state
            current_state = next_state

            # get the behavior at this step
            behavior = env.get_behavior(init_beh)
            trajectory.append(behavior)

        # get the behavior
        behavior = np.array(trajectory)
        behavior = get_equi_spaced_points(behavior, self.config["behavior"]["traj_length"])

        return episodic_return, behavior, stats

