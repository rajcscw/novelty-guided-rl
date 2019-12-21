import gym
import numpy as np
from gym import wrappers
import warnings
from gym import spaces
from enum import Enum
import gym_minigrid
from novelty_guided_package.environments.behaviors import get_behavoir_characteristic


class ActionSpaceType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class PolicyType(Enum):
    SOFTMAX = 0,
    GAUSSIAN = 1


class GymEnvironment:
    def __init__(self, name, max_episode_steps=None, render=False, save_loc=None):
        self.name = name
        self.max_episode_steps = max_episode_steps
        self.env = self._init_env(name, max_episode_steps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if render:
                self.env = wrappers.Monitor(self.env,
                                            directory=save_loc,
                                            force=True)

    def __repr__(self):
        return f"{self.name} environment with: \n {self.action_space_type} action space with {self.action_dim} actions \n Observation space with {self.state_dim} dim state"

    @staticmethod
    def _init_env(name, max_episode_steps):
        env = gym.make(name)
        if max_episode_steps:
            env.max_steps = max_episode_steps
            env._max_episode_steps = max_episode_steps
        return env

    @property
    def state_dim(self):
        env = GymEnvironment(self.name, self.max_episode_steps)
        return env.reset().shape[0]

    @property
    def action_dim(self):
        if isinstance(self.env.action_space, spaces.Discrete):
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    @property
    def action_space_type(self):
        return ActionSpaceType.CONTINUOUS if isinstance(self.env.action_space, spaces.Box) else ActionSpaceType.DISCRETE

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def policy_type(self):
        if self.action_space_type == ActionSpaceType.DISCRETE:
            return PolicyType.SOFTMAX
        elif self.action_space_type == ActionSpaceType.CONTINUOUS:
            return PolicyType.GAUSSIAN
        else:
            return None

    def reset(self):
        state = self.env.reset()
        state = state["image"].flatten() if isinstance(state, dict) else state.flatten()
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = state["image"].flatten() if isinstance(state, dict) else state.flatten()
        return state, reward, done, info

    def get_behavior(self, init_beh):
        return get_behavoir_characteristic(self.name, self.env, init_beh)


if __name__ == "__main__":
    print(GymEnvironment(name="MiniGrid-MultiRoom-N2-S4-v0", max_episode_steps=100))
    print(GymEnvironment(name="CartPole-v0", max_episode_steps=100))
    print(GymEnvironment(name="MountainCar-v0", max_episode_steps=100))
    print(GymEnvironment(name="MountainCarContinuous-v0", max_episode_steps=100))