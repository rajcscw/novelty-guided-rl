from itertools import product
import json


def get_dict_obj(keys, values):
    dict = {}
    for key, value in zip(keys, values):
        dict[key] = value
    return dict


def find_products(splits_by_keys):
    values = list(splits_by_keys.values())
    keys = list(splits_by_keys.keys())
    if len(values) == 1:
        dict_objs = [get_dict_obj(keys, [value]) for value in values[0]]
    else:
        product_values = product(*values)
        dict_objs = [get_dict_obj(keys, value) for value in product_values]
    return dict_objs


def split_config(obj):
    """
    Recursively splits the given object
    :return:
    """
    if not isinstance(obj, dict):
        return obj

    # it is a dict and further split
    splits_by_key = {}
    for key, value in obj.items():
        if isinstance(value, list):
            all_splits = []
            for item in value:
                splits = split_config(item)
                if isinstance(splits, list):
                    all_splits.extend(splits)
                else:
                    all_splits.append(splits)
            splits_by_key[key] = all_splits

        elif isinstance(value, dict):
            splits_by_key[key] = split_config(value)
        else:
            splits_by_key[key] = [value]

    # here, find cartesian
    configs = find_products(splits_by_key)

    return configs


if __name__ == "__main__":
    config = json.load(open("/home/rajkumar/IdeaProjects/novelty-guided-rl/scripts/experiments/configs/novelty_es.json"))
    configs = split_config(config)
    print(f" Total configs found: {len(configs)}")
    for config in configs:
        print(config)