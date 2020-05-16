import seaborn as sns
import pandas as pd
import os
import json
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from novelty_guided_package.core_components.utility import rolling_mean


############### BASELINE METHODS #######################
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
########################################################


def extract_params_seq(group_name):
    def extract_param_value(param_name):
        value = group_name.split(param_name+"=")[1].split("_")[0]
        return value

    hidden = extract_param_value("n_hidden")
    layers = extract_param_value("n_layers")
    sparsity_level = extract_param_value("sparsity_level")
    initial_rl_weight = extract_param_value("initial_rl_weight")
    novelty = True if initial_rl_weight == "0.0" else False
    adapt = True if extract_param_value("adapt") == "True" else False
    behavior_variable = True if extract_param_value("behavior_variable") == "True" else False
    return hidden, layers, sparsity_level, novelty, adapt, behavior_variable

def get_strategy_name_seq(group_name):
    hidden, layers, sparsity_level, novelty_or_adaptive, adapt_or_not, behavior_variable = extract_params_seq(group_name)
    if novelty_or_adaptive:
        if hidden == "nan" or sparsity_level == "nan":
            return "Novelty-KNN"
        else:
            return "Novelty-Seq-AE"
    else:
        if adapt_or_not:
            if hidden == "nan" or sparsity_level == "nan":
                return "Guided-KNN"
            else:
                return "Guided-Seq-AE"
        else:
            return "RL"

def to_pands_df_seq(results: Dict[str, List], 
                chosen_novelty_hidden: float, 
                chosen_novelty_layers: float, 
                chosen_novelty_sparsity: float,
                chosen_novelty_behavior_variable: float,
                chosen_adaptive_hidden: float,
                chosen_adaptive_layers: float,
                chosen_adaptive_sparsity: float,
                chosen_adaptive_behavior_variable: float, 
                chosen_rl_weight_delta: float,
                clip: float) -> pd.DataFrame:
    novelty_combined_df = pd.DataFrame()
    adaptive_combined_df = pd.DataFrame()
    for group_name, group_data in results.items():
        # get strategy name
        strategy_name = get_strategy_name_seq(group_name)
    
        for i, data in enumerate(group_data):
            
            # extract config from group name
            hidden, layers, sparsity_level, novelty_or_adaptive, adapt_or_not, behavior_variable = extract_params_seq(group_name)

            # clip the data
            if clip is not None:
                data = [clip if point >= clip else point for point in data]

            df = pd.DataFrame()
            df["epoch"] = np.arange(len(data))
            df["episodic_total_rewards"] = rolling_mean(data, 200) # apply some smoothing here
            df["run"] = i
            df["strategy"] = strategy_name

            # let's create separate plots for novelty
            if novelty_or_adaptive and (hidden in [chosen_novelty_hidden, "nan"]) and sparsity_level in [chosen_novelty_sparsity, "nan"] and \
                layers == chosen_novelty_layers and behavior_variable == chosen_novelty_behavior_variable:
                novelty_combined_df = novelty_combined_df.append(df)

            # let's create separate plots for adaptive
            if not novelty_or_adaptive and (hidden in [chosen_adaptive_hidden, "nan"]) and \
                sparsity_level in [chosen_adaptive_sparsity, "nan"] and \
                chosen_adaptive_layers == layers and behavior_variable == chosen_adaptive_behavior_variable:
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
    plt.ylabel("Episodic Total Reward", fontsize=20)
    plt.xlabel("Steps", fontsize=20)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=15)

    labels = list(results_df["strategy"].unique())
    legend = []
    for style_ in style:
        if style_ in labels:
            legend.append(style_)

    plt.legend(loc=4, fontsize=15, labels=legend)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# path to the results folder
path_to_results = os.path.join(os.path.dirname(os.path.realpath(__file__)), "seq_results")
path_to_baseline_results = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
path_to_outputs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plot_outputs_icann")

# tasks to analyze
tasks = [
    "half_cheetah",
    "hopper",
    # "walker",
    "inverted_pendulum",
    "inverted_double_pendulum"
]

# best settings for each task (found through dashify analysis)
tasks_and_settings = {
    # novelty and adaptive
    "half_cheetah": (("50", "2", "1.0", True), ("50", "2", "0.5", True), None),
    "inverted_pendulum": (("50", "2", "0.5", False), ("50", "2", "0.25", True), 5000),
    "inverted_double_pendulum": (("50", "2", "0.25", False), ("50", "2", "0.25", False), None),
    "hopper": (("100", "1", "0.25", True), ("100", "1", "0.5", True), None),
}

# for the baseline methods
# best settings for each task (found through dashify analysis) (more)
baseline_tasks_and_settings = {
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

    # load the corresponding json for baseline
    baseline_result_file = os.path.join(path_to_baseline_results, f"{task}.json")
    baseline_results = json.load(open(baseline_result_file)) 

    # chosen rl weight delta
    chosen_rl_weight_delta = "0.05"

    task_folder = os.path.join(path_to_outputs)
    os.makedirs(task_folder, exist_ok=True)

    # to a dataframe
    novelty_results_df, adaptive_results_df = to_pands_df_seq(results, 
                                                           tasks_and_settings[task][0][0], 
                                                           tasks_and_settings[task][0][1],
                                                           tasks_and_settings[task][0][2],
                                                           tasks_and_settings[task][0][3],
                                                           tasks_and_settings[task][1][0], 
                                                           tasks_and_settings[task][1][1],
                                                           tasks_and_settings[task][1][2],
                                                           tasks_and_settings[task][1][3],
                                                           chosen_rl_weight_delta,
                                                           tasks_and_settings[task][2])

    # baselines - to a dataframe
    baseline_novelty_results_df, baseline_adaptive_results_df = to_pands_df(baseline_results, 
                                                                         baseline_tasks_and_settings[task][0][0], 
                                                                         baseline_tasks_and_settings[task][0][1],
                                                                         baseline_tasks_and_settings[task][1][0], 
                                                                         baseline_tasks_and_settings[task][1][1],
                                                                         chosen_rl_weight_delta)

    # concatenate the results
    novelty_results_df = pd.concat([novelty_results_df, baseline_novelty_results_df])
    adaptive_results_df = pd.concat([adaptive_results_df, baseline_adaptive_results_df])

    # plot the figure
    plot_df(f"{task_folder}/{task}_novelty.pdf", novelty_results_df, ["Novelty-KNN", "ES", "Novelty-AE", "Novelty-Seq-AE"], sns.color_palette("Set1", n_colors=4))
    plot_df(f"{task_folder}/{task}_adaptive.pdf", adaptive_results_df, ["Guided-KNN", "RL", "Guided-AE", "Guided-Seq-AE"], sns.color_palette("Set1", n_colors=4))