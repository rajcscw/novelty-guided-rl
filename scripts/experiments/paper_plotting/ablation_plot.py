import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import os
import json
from typing import List, Dict 
import seaborn as sns

def extract_param_value(group_name, param_name):
    value = group_name.split(param_name+"=")[1].split("_")[0]
    return value

def plot(file_name: str, results: Dict[str, List], selected_method_name: str, param_name: str, strategy_id: str):

    # collect the results as df
    combined_df = pd.DataFrame() 
    for group_name, group_data in results.items():
        method_name = extract_param_value(group_name, "name")
        if selected_method_name != method_name:
            continue
        sequence_length = extract_param_value(group_name, param_name)
        for i, data in enumerate(group_data):
            df = pd.DataFrame()
            df["epoch"] = [1]
            df["Final Performance"] = np.mean(data[:-100])
            df[strategy_id] = int(sequence_length)
            combined_df = combined_df.append(df)
    
    # order
    order = sorted(combined_df[strategy_id].unique())

    print(order)

    plt.figure()
    fig = plt.figure()
    sns.set(style="darkgrid")
    sns.set_context("paper")
    sns.boxplot(x=strategy_id, 
                y="Final Performance", 
                palette=sns.color_palette("Blues", n_colors=3),
                data=combined_df, 
                width=0.4,
                linewidth=2.5,
                order=order)
    plt.ylabel("Total Reward", fontsize=20)
    plt.xlabel(strategy_id, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
             
        
# path to the results folder
path_to_results = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
path_to_outputs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "plot_outputs")


for item in ["sequence_length", "archive_size"]:
    # load the corresponding json
    result_file = os.path.join(path_to_results, f"{item}.json")
    results = json.load(open(result_file)) 

    if item == "sequence_length":
        param_name = "behavior_traj_length"
        strategy_id = "Sequence Length"
    else:
        param_name = "archive_size"
        strategy_id = "Archive Size"

    # plot the results
    plot(os.path.join(path_to_outputs, f"{item}_ablation_ae.pdf"), results, "ae", param_name, strategy_id)
    plot(os.path.join(path_to_outputs, f"{item}_ablation_knn.pdf"), results, "knn", param_name, strategy_id)