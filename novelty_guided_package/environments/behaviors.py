import numpy as np

"""
Behavior Characteristic is at the moment hard coded
"""


def get_behavoir_characteristic(env_name, env, init_beh=None):
    # 2-D environments
    if env_name in ["Hopper-v2", "Swimmer-v2", "Walker2d-v2", "HalfCheetah-v2"]:
        behavior = np.array(_get_pos(env.env)[0:2])
        if init_beh is not None:
            x_pos_now = behavior[0]
            x_pos_init = init_beh[0]
            if np.sign(x_pos_now - x_pos_init) < 0.0:  # clip the behavior space
                behavior[0] = x_pos_init
            offset = (behavior - init_beh) ** 2
            return offset
        else:
            return behavior
    # 3-D environments
    elif env_name in ["Humanoid-v2", "HumanoidStandup-v2"]:
        behavior = np.array(_get_pos(env.env)[0:3])
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
    elif "InvertedPendulum-v2" == env_name:
        cart_pos = env.env.get_body_com("cart")[:2]
        pole_pos = env.env.get_body_com("pole")[:2]
        return np.concatenate([cart_pos, pole_pos])
    elif "InvertedDoublePendulum-v2" == env_name:
        cart_pos = env.env.get_body_com("cart")[:2]
        pole_pos = env.env.get_body_com("pole")[:2]
        pole_pos2 = env.env.get_body_com("pole2")[:2]
        return np.concatenate([cart_pos, pole_pos, pole_pos2])
    # Reacher environments
    elif "Reacher-v2" == env_name:
        feedback_signal = (env.env.get_body_com("fingertip") - env.env.get_body_com("target"))[0:2]
        return feedback_signal
    # for mini grid environments
    elif "Grid" in env_name:
        if init_beh is not None:
            behavior = np.array(env.agent_pos)
            offset = (behavior - init_beh)
            return offset
        else:
            # just get the agent pos
            behavior = np.array(env.agent_pos)
            return behavior


def _get_pos(env):
    mass = env.model.body_mass.reshape(-1,1)
    xpos = env.data.xipos
    center = (np.sum(mass * xpos, 0) / np.sum(mass))
    return center[0], center[1], center[2]