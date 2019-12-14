import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# set the backend
plt.switch_backend('agg')


def rolling_mean(X, window_size, pad=None):
    # pad in the front
    if pad is None:
        front = np.full((window_size,), X[0]).tolist()
    else:
        front = np.full((window_size,), pad).tolist()
    padded = front + X
    mean = np.convolve(padded, np.ones((window_size,))/window_size, "valid")
    return mean


def plot_learning_curve(file, title, series):
    plt.figure()
    fig = plt.figure()
    sns.set(style="darkgrid")
    sns.set_context("paper")
    sns.lineplot(data=series, x="Steps", hue="strategy", y="Episodic Total Reward", err_style="band", style="strategy")
    plt.title(title, fontsize=10)
    plt.ylabel("Episodic Total Reward", fontsize=10)
    plt.xlabel("Steps", fontsize=10)
    plt.legend(loc=4, fontsize=10)
    plt.savefig(file)


def to_device(tensor, device):
    if device == "cpu":
        return tensor
    else:
        return tensor.cuda(device)


def init_multiproc():
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
