from itertools import product
from copy import deepcopy
from typing import List


def find_lists_in_dict(obj_dict, param_grid: List) -> List:
    """
    Traverses the possibly nested dictionaries and if a leaf (-> a value) is of type List,
    then this leaf is added to the param_grid list as a sublist.
    :param obj_dict:
    :param param_grid:
    :return:
    """
    for key in obj_dict:
        if isinstance(obj_dict[key], list):
            param_grid.append(obj_dict[key])
        elif isinstance(obj_dict[key], dict):
            find_lists_in_dict(obj_dict[key], param_grid)
        else:
            continue
    return param_grid


def replace_lists_in_dict(obj, obj_copy, comb, counter):
    for key, key_copy in zip(obj, obj_copy):
        if isinstance(obj[key], list):
            obj_copy[key_copy] = comb[len(counter)]
            counter.append(1)
        elif isinstance(obj[key], dict):
            replace_lists_in_dict(obj[key], obj_copy[key_copy], comb, counter)
        else:
            continue
    return obj_copy, counter


def split_gs_config(config_grid_search):
    param_grid = []
    param_grid = find_lists_in_dict(config_grid_search, param_grid)
    config_copy = deepcopy(config_grid_search)
    for comb in product(*param_grid):
        counter = []
        individual_config = replace_lists_in_dict(config_grid_search, config_copy, comb, counter)[0]
        individual_config = deepcopy(individual_config)
        yield individual_config