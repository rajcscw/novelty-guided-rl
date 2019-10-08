import numpy as np
from datetime import datetime
import os
import yaml
import torch
dir_path = os.path.dirname(os.path.realpath(__file__))
import pandas as pd
from components.core_components.utility import init_multiproc, plot_learning_curve, rolling_mean
from components.seq2seq_components.Seq2SeqNovelty import NoveltyDetectionModule as AEModel
import sys
import gym.spaces
from components.core_components.run_utility import run_experiment_gym_with_es
import pickle
from gym_minigrid.register import env_list
from components.core_components.run_baseline import run_experiment_PPO
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from copy import deepcopy


def get_task_configs(task_names):

    task_configs = {}
    for task_name in task_names:
        # read the corresponding config
        with open(dir_path+"/configs/{}.yml".format(task_name), 'r') as ymlfile:
            config = yaml.load(ymlfile)

        # get the individial task names
        names = config["environment"]["name"]

        # now, replicate the config
        for name in names:
            task_config = deepcopy(config)
            task_config["environment"]["name"] = name
            task_configs[name] = task_config
    return task_configs


if __name__ == "__main__":

    init_multiproc()
    # import very late here is required so that start method is set to spawn
    # this is basically required so that CUDA can work with multiprocessing
    from components.baseline_components.KNNNovelty import NoveltyDetectionModule as KNNModel

    # check the device (CPU vs GPU)
    use_gpu = False
    device_count = torch.cuda.device_count()
    device_list = list(map(lambda x: "cuda:{}".format(x), range(device_count))) if torch.cuda.is_available() and use_gpu else ["cpu"]

    print("\n The current device is: "+str(device_list))

    # read the arguments for the task to execute
    if len(sys.argv) > 1:

        task_names = sys.argv[1:]

        # get the individual task names from the config
        task_names_and_configs = get_task_configs(task_names)

        for task_name, config in task_names_and_configs.items():

            print("\nRunning for task name:{}".format(task_name))

            # create folder name
            os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/outputs/", exist_ok=True)
            folder_name = os.path.dirname(os.path.realpath(__file__)) + "/outputs/Comparison-all-"+str(datetime.now())
            os.mkdir(folder_name)

            # data frame to hold all results
            episodic_total_rewards_strategy = pd.DataFrame()

            for adaptive_config in [
                                    #None,
                                    #False,
                                    True,
                                    ]:
                # None - indicates it is purely RL driven
                # False - indicates it is purely Novelty driven
                # True - indicates it is combination of RL and Novelty driven

                # create environments to get
                env = gym.make(config["environment"]["name"])

                # create environments to get
                if config["environment"]["mini_grid"]:
                    env = gym.make(config["environment"]["name"])
                    action_dim = env.action_space.n
                    state_dim = 7 * 7 * 3 # hard-coded for now
                else: # for other continuous control tasks
                    env = gym.make(config["environment"]["name"])
                    action_dim = env.action_space.shape[0]
                    state_dim = env.observation_space

                # different configurations for novelty
                novelty_configs = {
                    #"KNN": (KNNModel(k=config["NNnovelty"]["k"], limit=config["NNnovelty"]["limit"]), "knn_trajectory"),
                    "AE": (AEModel(input_dim=config["Seqnovelty"]["input_dim"],
                                   hidden_size=config["Seqnovelty"]["hidden_size"],
                                   n_layers=config["Seqnovelty"]["n_layers"],
                                   device=device_list[0],
                                   lr=float(config["Seqnovelty"]["lr"]),
                                   reg=float(config["Seqnovelty"]["reg"])), "ae_trajectory",),
                }

                # it is not none
                if adaptive_config is not None:
                    # iterate over different novelty configs
                    for novelty_name, novelty_param in novelty_configs.items():

                        # params
                        novelty_detector = novelty_param[0]
                        behavoir = novelty_param[1]

                        # adaptive or not
                        if adaptive_config: # start with pure RL reward and adapt pressure accordingly
                            rl_weight = 1.0
                            strategy_name = "NSRA-"+novelty_name
                        else: # if else, use the defined rl weight from the config - it is 0.0 for novelty
                            rl_weight = config["novelty"]["initial_rl_weight"]
                            strategy_name = "NS-"+novelty_name

                        for k in range(int(config["log"]["runs"])):

                            print("\n -------Running strategy {} for iteration {} ----------".format(strategy_name, k+1))

                            # create a directory
                            os.mkdir(folder_name+"/"+strategy_name+str(k))

                            # behavior file
                            behavoir_file = None

                            # run the experiment once
                            episodic_total_reward = run_experiment_gym_with_es(env, config, state_dim, action_dim, rl_weight, adaptive_config, folder_name + "/" + strategy_name + str(k), novelty_detector, behavoir, device_list, behavoir_file)

                            # store raw total rewards (unsmoothened)
                            with open(folder_name+"/"+strategy_name+str(k)+"/raw_episodic_total_rewards.p", "wb") as f:
                                pickle.dump(episodic_total_reward, f)

                            # Compute the running mean
                            N = int(config["log"]["average_every"])
                            selected = rolling_mean(episodic_total_reward, N, config["log"]["initial_pad_value"]).tolist()

                            # Combine all runs and add them to the strategy dataframe
                            df = pd.DataFrame({"Steps": np.arange(len(selected)), "Episodic Total Reward": selected})
                            df["run"] = k
                            df["strategy"] = strategy_name
                            episodic_total_rewards_strategy = episodic_total_rewards_strategy.append(df)

                else:
                    # pure RL driven
                    rl_weight = 1.0
                    strategy_name = "RL"

                    for k in range(int(config["log"]["runs"])):

                        print("\n -------Running strategy {} for iteration {} ----------".format(strategy_name, k+1))

                        # create a directory
                        os.mkdir(folder_name+"/"+strategy_name+str(k))

                        # behavior file
                        behavoir_file = None

                        # run the experiment once
                        episodic_total_reward = run_experiment_gym_with_es(env, config, state_dim, action_dim, rl_weight, False, folder_name + "/" + strategy_name + str(k), None, None, device_list, behavoir_file)

                        # store raw total rewards (unsmoothened)
                        with open(folder_name+"/"+strategy_name+str(k)+"/raw_episodic_total_rewards.p", "wb") as f:
                            pickle.dump(episodic_total_reward, f)

                        # Compute the running mean
                        N = int(config["log"]["average_every"])
                        selected = rolling_mean(episodic_total_reward, N, config["log"]["initial_pad_value"]).tolist()

                        # Combine all runs and add them to the strategy dataframe
                        df = pd.DataFrame({"Steps": np.arange(len(selected)), "Episodic Total Reward": selected})
                        df["run"] = k
                        df["strategy"] = strategy_name
                        episodic_total_rewards_strategy = episodic_total_rewards_strategy.append(df)

            # # also get the policy gradient baseline (PPO) (run for the same number of times)
            # for k in range(int(config["log"]["runs"])):
            #
            #     print("\n -------Running PPO for iteration----------"+ str(k+1))
            #
            #     strategy_name = "PPO"
            #     os.mkdir(folder_name+"/"+strategy_name+str(k))
            #     config_ppo = config["PPO"]
            #     config_ppo["nEps"] = config["log"]["parallel_eval"]
            #     config_ppo["nItrs"] = config["log"]["iterations"]
            #     _, episodic_total_reward = run_experiment_PPO(config["environment"]["name"], config["PPO"])
            #
            #     # Compute the running mean
            #     N = int(config["log"]["average_every"])
            #     selected = rolling_mean(episodic_total_reward, N, config["log"]["initial_pad_value"]).tolist()
            #
            #     # Combine all runs and add them to the strategy dataframe
            #     df = pd.DataFrame({"Steps": np.arange(len(selected)), "Episodic Total Reward": selected})
            #     df["run"] = k
            #     df["strategy"] = strategy_name
            #     episodic_total_rewards_strategy = episodic_total_rewards_strategy.append(df)

            # Plot the learning curves here
            episodic_total_rewards_strategy.to_pickle(folder_name+"/learning_curve_df")
            plot_learning_curve(folder_name+"/"+config["environment"]["name"]+"_Episodic_total_reward.pdf", config["environment"]["name"], episodic_total_rewards_strategy)

            # Log all the config parameters
            file = open(folder_name+"/model_config.txt", "w")
            file.write(str(config))
            file.close()

            # Let's all save the task name (to be later used for visualization)
            pickle.dump(config["environment"]["name"], open(folder_name+"/task_name.p", "wb"))

    else:
        print("\n Enter the task name to be experimented...")