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
    return mean[:len(X)].tolist()


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


def get_fixed_length_sequences(seq, to_len):
    seq_len = seq.shape[0]

    if seq_len == 0:
        return np.zeros((to_len, 2))

    space = int(np.ceil(seq_len/to_len))
    if space > 0:
        spaced_indices = np.arange(0, seq_len - 1, space)
        if len(spaced_indices) < to_len:
            # then pad
            padded = np.zeros(to_len-spaced_indices.shape[0], dtype=np.int8).flatten()
            spaced_indices = np.concatenate((padded, spaced_indices))
    else:
        spaced_indices = np.arange(0, seq_len)
    assert spaced_indices.shape[0] == to_len
    sampled = seq[spaced_indices]
    return sampled

def get_variable_length_sequences(seq, sampling_ratio):
    seq_len = seq.shape[0]
    dim = seq.shape[1]
    sampling_rate = sampling_ratio
    
    if seq_len < sampling_rate:
        # let's consider sampling ratio as the sampling rate
        sampling_rate = int(sampling_rate / 10)
        spaced_indices = np.arange(sampling_rate, seq_len+1, sampling_rate)
        spaced_indices = spaced_indices - 1
        spaced_indices_ = [0]
        spaced_indices_.extend(spaced_indices.tolist())
    else:
        # let's consider sampling ratio as the sampling rate
        spaced_indices = np.arange(sampling_rate, seq_len+1, sampling_rate)
        spaced_indices = spaced_indices - 1
        spaced_indices_ = [0]
        spaced_indices_.extend(spaced_indices.tolist())
    seq = seq[spaced_indices_]
    return seq

if __name__ == "__main__":
    sampling_rate = 100
    seq_length = 20
    seq = np.random.rand(seq_length, 1)
    print(get_variable_length_sequences(seq, None, None, sampling_rate).shape)