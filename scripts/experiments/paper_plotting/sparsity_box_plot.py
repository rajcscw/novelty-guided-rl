import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List
from tqdm import tqdm

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


def to_pands_df(results: Dict[str, List], chosen_layer_config) -> pd.DataFrame:
    novelty_combined_df = pd.DataFrame()
    adaptive_combined_df = pd.DataFrame()
    for group_name, group_data in results.items():    
        for i, data in enumerate(group_data):
            
            # extract config from group name
            hidden, sparsity_level, novelty_or_adaptive, adapt_or_not, rl_weight_delta = extract_params(group_name)

            if hidden not in chosen_layer_config:
                continue

            df = pd.DataFrame()
            df["epoch"] = np.arange(1)
            df["Total Reward"] = np.mean(data[:-100])
            df["run"] = i
            df["Layer Configuration"] = hidden
            df["Sparsity Level"] = str(sparsity_level)

            # let's create separate plots for novelty
            if novelty_or_adaptive and (hidden != "nan") and (sparsity_level != "nan"):
                novelty_combined_df = novelty_combined_df.append(df)

            if not novelty_or_adaptive and (hidden != "nan") and sparsity_level != "nan":
                adaptive_combined_df = adaptive_combined_df.append(df)
                
    return novelty_combined_df, adaptive_combined_df


def plot_df( file_name: str, df: pd.DataFrame, x: str, y: str, palette: str, hue: str):
    plt.figure()
    fig = plt.figure()
    sns.set(style="darkgrid")
    sns.set_context("paper")
    sns.boxplot(x=x, y=y, palette=palette,data=df, hue=hue, hue_order=["0.25", "0.5", "1.0"], width=0.3, linewidth=2.5)
    plt.ylabel(y, fontsize=20)
    plt.xlabel(x, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# tasks to analyze
tasks = [
 "half_cheetah",
 "hopper",
 "walker",
 "inverted_pendulum",
 "inverted_double_pendulum"  
]

# path to the results folder
path_to_results = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
path_to_outputs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plot_outputs/sparsity_plots")
os.makedirs(path_to_outputs, exist_ok=True)

# analyze and produce output for each task
for task in tqdm(tasks):
    # load the corresponding json
    result_file = os.path.join(path_to_results, f"{task}.json")
    results = json.load(open(result_file)) 

    # to a dataframe
    novelty_results_df, adaptive_results = to_pands_df(results, chosen_layer_config=["100", "50"])

    plot_df(f"{path_to_outputs}/{task}_sparsity_novelty.pdf", novelty_results_df, "Layer Configuration", "Total Reward", sns.color_palette("Blues", n_colors=3), "Sparsity Level")
    plot_df(f"{path_to_outputs}/{task}_sparsity_adaptive.pdf", adaptive_results, "Layer Configuration", "Total Reward", sns.color_palette("Blues", n_colors=3), "Sparsity Level")