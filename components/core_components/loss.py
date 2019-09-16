import numpy as np
import gym
import warnings
gym.logger.set_level(40)
from gym import spaces
from gym import wrappers
import torch
from torch.nn import Softmax


class EpisodicReturnPolicy:
    def __init__(self, model, config, scaler, state_dim, action_dim, behavior_selection=None, is_deterministic = False, eval=1):
        self.model = model
        self.config = config
        self.scaler = scaler
        self.behavior_selection = behavior_selection
        self.n_sv = state_dim
        self.n_a = action_dim
        self.is_deterministic = is_deterministic
        self.eval = eval
        self.policy_type = config["policy"]["type"]

        # dummy variables to hold current parameter name
        self.parameter_name = None
        self.add_noise = False

        # Set up the environment
        self.env_params = self.config["environment"]

        # Set up the model
        self.model = model

        # max steps
        self.max_steps = config["environment"]["max_episode_steps"] if "max_episode_steps" in config["environment"].keys() else None

    def get_behavoir_characteristic(self, env, init_beh=None):
        # 2-D environments
        if self.env_params["name"] in ["Hopper-v2", "Swimmer-v2", "Walker2d-v2", "HalfCheetah-v2"]:
            behavior = np.array(self._get_pos(env.env)[0:2])
            if init_beh is not None:
                x_pos_now = behavior[0]
                x_pos_init = init_beh[0]
                if np.sign(x_pos_now - x_pos_init) < 0.0:  # clip the behavior space
                    behavior[0] = x_pos_init
                offset = (behavior - init_beh) ** 2
                return offset
            else:
                return behavior
        # Pendulum environments
        elif "InvertedPendulum-v2" == self.env_params["name"]:
            cart_pos = env.env.get_body_com("cart")[:2]
            pole_pos = env.env.get_body_com("pole")[:2]
            return np.concatenate([cart_pos, pole_pos]), np.concatenate([cart_pos, pole_pos])
        elif "InvertedDoublePendulum-v2" == self.env_params["name"]:
            cart_pos = env.env.get_body_com("cart")[:2]
            pole_pos = env.env.get_body_com("pole")[:2]
            pole_pos2 = env.env.get_body_com("pole2")[:2]
            return np.concatenate([cart_pos, pole_pos, pole_pos2]), np.concatenate([cart_pos, pole_pos])
        # for mini grid environments
        elif "Grid" in self.env_params["name"]:
            if init_beh is not None:
                behavior = np.array(env.agent_pos)
                offset = (behavior - init_beh)
                return offset
            else:
                # just get the agent pos
                behavior = np.array(env.agent_pos)
                return behavior

    def get_equi_spaced_points(self, seq, to_len):
        seq_len = seq.shape[0]
        space = int(np.ceil(seq_len/to_len))
        if space > 0:
            spaced_indices = np.arange(0, seq_len - 1, space)
            if len(spaced_indices) < to_len:
                # then pad
                padded = np.zeros(to_len-spaced_indices.shape[0], dtype=np.int8).flatten()
                spaced_indices = np.concatenate((padded, spaced_indices))
        else:
            spaced_indices = np.arange(0, seq_len)
        assert spaced_indices.shape[0] == to_len
        sampled = seq[spaced_indices]
        return sampled

    def get_best_action(self, env, state, device):

        # transform
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.scaler is not None:
                state = self.scaler.transform(state.reshape((1,-1))).flatten()

        # choose the action
        state = torch.from_numpy(state.flatten()).type(torch.float32).view(1,-1)

        # send it to the device
        state = self.to_device(state, device)

        network_output = self.model.net.forward(state)

        if self.policy_type == "gaussian":
            # apply gaussian policy
            action = self.gaussian_policy(network_output)

            # apply the bounds
            a_min = env.action_space.low.reshape((-1,1))
            a_max = env.action_space.high.reshape((-1,1))
            action = np.clip(action, a_min, a_max)
        elif self.policy_type == "softmax":
            # apply softmax
            action = self.softmax_policy(network_output)

        return action

    def softmax_policy(self, network_output):
        network_output = Softmax(dim=1)(network_output)
        action = torch.multinomial(input=network_output, num_samples=1).cpu().data.numpy()
        #action = torch.argmax(input=network_output).cpu().data.numpy()
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

    def _get_pos(self, env):
        mass = env.model.body_mass.reshape(-1,1)
        xpos = env.data.xipos
        center = (np.sum(mass * xpos, 0) / np.sum(mass))
        return center[0], center[1], center[2]

    def _get_body_com(self, env):
        pos = env.get_body_com("torso")
        return pos

    def __call__(self, parameter_value, device, eval=False, render=False, save_loc=None):
        # create the environment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = gym.make(self.env_params["name"])
            if self.env_params["mini_grid"]:
                env = env
            if self.max_steps:
                env._max_episode_steps = self.max_steps

            if render:
                env = wrappers.Monitor(env,
                                       directory=save_loc,
                                       force=True)

        # build a model with this parameter
        if parameter_value is not None:
            self.model.set_layer_value(self.parameter_name, parameter_value)

        # pass the model to the device
        self.model.net = self.to_device(self.model.net, device)

        # now we actually play an episode with the environment
        current_state = env.reset()

        # get the init beh
        if render:
            init_beh = self.get_behavoir_characteristic(env.env, None)
        else:
            init_beh = self.get_behavoir_characteristic(env, None)

        episodic_return = 0
        stats = []
        trajectory = []
        steps_taken = 0
        while True:
            # Render the environment
            if render:
                env.render(mode="rgb_array")

            if self.env_params["mini_grid"]:
                current_state = current_state["image"].flatten()

            # Choose the best action
            current_action = self.get_best_action(env, current_state, device)

            # Perform the chosen action and get the reward and next state of the environment
            next_state, current_reward, done, info = env.step(current_action.flatten())
            stats.append(info)
            steps_taken += 1
            episodic_return += current_reward

            if done or (self.max_steps is not None and steps_taken >= self.max_steps):
                break

            # Set the next state
            current_state = next_state

            # get the behavior at this step
            if render:
                behavior = self.get_behavoir_characteristic(env.env, init_beh)
            else:
                behavior = self.get_behavoir_characteristic(env, init_beh)
            trajectory.append(behavior)

        # get the final position of the agent
        if self.behavior_selection == "knn_trajectory":
            behavior = np.array(trajectory)
            behavior = self.get_equi_spaced_points(behavior, self.config["NNnovelty"]["traj_length"])
        elif self.behavior_selection == "ae_trajectory": # a sequence of agent#s trajectory
            behavior = np.array(trajectory)

            # let's apply variable length sequences
            # that is we subsample the sequences
            if self.config["Seqnovelty"]["variable_length"]:
                if behavior.shape[0] < self.config["Seqnovelty"]["min_traj_length"]:
                    traj_length = self.config["Seqnovelty"]["min_traj_length_subsampled"]
                else:
                    traj_length = int(behavior.shape[0] / self.config["Seqnovelty"]["traj_sample_ratio"])
            else:
                traj_length = self.config["Seqnovelty"]["fixed_traj_length"]
            behavior = self.get_equi_spaced_points(behavior, traj_length)
            behavior = torch.from_numpy(behavior)
        else:
            behavior = None

        return episodic_return, behavior, stats

    @staticmethod
    def to_device(tensor, device):
        if device == "cpu":
            return tensor
        else:
            return tensor.cuda(device)

