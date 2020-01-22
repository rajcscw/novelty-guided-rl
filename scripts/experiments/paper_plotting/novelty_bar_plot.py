import seaborn as sns
import pandas as pd
import os
import json
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from novelty_guided_package.core_components.utility import rolling_mean

def extract_params(group_name):
    def extract_param_value(param_name):
        value = group_name.split(param_name+"=")[1].split("_")[0]
        return value

    hidden = extract_param_value("n_hidden")
    sparsity_level = extract_param_value("sparsity_level")
    initial_rl_weight = extract_param_value("initial_rl_weight")
    novelty = True if initial_rl_weight == "0.0" else False
    adapt = True if extract_param_value("adapt") == "True" else False
    rl_weight_delta = extract_param_value("rl_weight_delta")

    return hidden, sparsity_level, novelty, adapt, rl_weight_delta

def get_strategy_name(group_name):
    hidden, sparsity_level, novelty_or_adaptive, adapt_or_not, rl_weight_delta = extract_params(group_name)
    if novelty_or_adaptive:
        if hidden == "nan" or sparsity_level == "nan":
            return "Novelty-KNN"
        else:
            return "Novelty-AE"
    else:
        if adapt_or_not:
            if hidden == "nan" or sparsity_level == "nan":
                return "Guided-KNN"
            else:
                return "Guided-AE"
        else:
            return "RL"

def to_pands_df(results: Dict[str, List], 
                chosen_novelty_hidden: float, 
                chosen_novelty_sparsity: float,
                chosen_adaptive_hidden: float,
                chosen_adaptive_sparsity: float,  
                chosen_rl_weight_delta: float) -> pd.DataFrame:
    novelty_combined_df = pd.DataFrame()
    adaptive_combined_df = pd.DataFrame()
    for group_name, group_data in results.items():
        # get strategy name
        strategy_name = get_strategy_name(group_name)
    
        for i, data in enumerate(group_data):
            
            # extract config from group name
            hidden, sparsity_level, novelty_or_adaptive, adapt_or_not, rl_weight_delta = extract_params(group_name)

            df = pd.DataFrame()
            df["epoch"] = np.arange(len(data))
            df["episodic_total_rewards"] = rolling_mean(data, 200) # apply some smoothing here
            df["run"] = i
            df["strategy"] = strategy_name

            # let's create separate plots for novelty
            if novelty_or_adaptive and (hidden in [chosen_novelty_hidden, "nan"]) and sparsity_level in [chosen_novelty_sparsity, "nan"]:
                novelty_combined_df = novelty_combined_df.append(df)

            # let's create separate plots for adaptive
            if not novelty_or_adaptive and (hidden in [chosen_adaptive_hidden, "nan"]) and \
                sparsity_level in [chosen_adaptive_sparsity, "nan"] and rl_weight_delta in [chosen_rl_weight_delta, "nan"]:
                adaptive_combined_df = adaptive_combined_df.append(df)
                
    return novelty_combined_df, adaptive_combined_df

def plot_df(file_name, results_df, style, pal):
    plt.figure()
    fig = plt.figure()
    sns.set(style="darkgrid")
    sns.set_context("paper")
    sns.lineplot(x="epoch", 
                 y="episodic_total_rewards", 
                 hue="strategy", 
                 data=results_df, 
                 ci="sd",
                 hue_order=style,
                 err_style="band", 
                 palette=pal,
                 )
    plt.ylabel("Episodic Total Reward", fontsize=10)
    plt.xlabel("Steps", fontsize=10)

    labels = list(results_df["strategy"].unique())
    legend = []
    for style_ in style:
        if style_ in labels:
            legend.append(style_)

    plt.legend(loc=4, fontsize=10, labels=legend)
    plt.savefig(file_name)
    plt.close()

# path to the results folder
path_to_results = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
path_to_outputs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plot_outputs")

# tasks to analyze
tasks = [
    #"half_cheetah",
    #"hopper",
    #"walker",
    #"inverted_pendulum",
    "inverted_double_pendulum"
]

# best settings for each task (found through dashify analysis) (more)
tasks_and_settings = {
    # TASK: (novelty, adaptive)
    "half_cheetah": (("100", "0.25"), ("100,50", "0.5")),
    "hopper": (("100,50", "0.25"), ("100", "0.25")),
    "walker": (("100,50", "1.0"), ("100,50", "0.5")),
    "inverted_pendulum": (("100,50", "1.0"), ("50", "0.25")),
    "inverted_double_pendulum": (("100", "0.25"), ("200, 100", "1.0"))
}

# analyze and produce output for each task
for task in tqdm(tasks):

    # load the corresponding json
    result_file = os.path.join(path_to_results, f"{task}.json")
    results = json.load(open(result_file)) 

    # loop over rl_weight_deltas to plot separately
    chosen_rl_weight_deltas = ["0.02", "0.05"]

    for chosen_rl_weight_delta in chosen_rl_weight_deltas:

        # task folder
        task_folder = os.path.join(path_to_outputs, f"with_{chosen_rl_weight_delta}")
        os.makedirs(task_folder, exist_ok=True)

        # to a dataframe
        novelty_results_df, adaptive_results = to_pands_df(results, 
                                                           tasks_and_settings[task][0][0], 
                                                           tasks_and_settings[task][0][1],
                                                           tasks_and_settings[task][1][0], 
                                                           tasks_and_settings[task][1][1],
                                                           chosen_rl_weight_delta)

        # plot the figure
        plot_df(f"{task_folder}/{task}_novelty.pdf", novelty_results_df, ["Novelty-KNN", "ES", "Novelty-AE"], sns.color_palette("Set1", n_colors=3))
        plot_df(f"{task_folder}/{task}_adaptive.pdf", adaptive_results, ["Guided-KNN", "RL", "Guided-AE"], sns.color_palette("Set1", n_colors=3))