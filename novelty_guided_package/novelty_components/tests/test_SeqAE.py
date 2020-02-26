import numpy as np
from novelty_guided_package.novelty_components.SeqAENovelty import SeqAE
import torch
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

# generate a time series (Mackey-Glass)
file_dir = os.path.dirname(os.path.realpath(__file__))
data = np.loadtxt(os.path.join(file_dir, "MackeyGlass_t17.txt"))

# generate a batch of sequences
n_samples = 1000
min_length = 50
max_length = 100
lengths = np.random.randint(min_length, max_length, n_samples)
indices = np.random.randint(0, data.shape[0]-max_length, n_samples)
all_sequences = []
for index, length in zip(indices, lengths):
    all_sequences.append(torch.from_numpy(data[index:index+length].reshape(-1,1)))

# novelty detection module (auto-encoder basically here)
batch_size = 100
autoencoder = SeqAE(n_input=1, n_hidden=50, n_layers=2, lr=1e-2, batch_size=batch_size, sparsity_level=1.0, device="cpu")

# run for many iterations
loss_list = []
running_loss = 0
max_iters = 500
for i in tqdm(range(max_iters)):
    # train it now
    loss = autoencoder.train(all_sequences)
    running_loss += loss

    # reconstrcuted
    random_idx = np.random.choice(range(n_samples))
    random_sequence = all_sequences[random_idx]
    reconstructed, _, _ = autoencoder.forward([random_sequence])

    if (i+1) % 20 == 0:
        print("Iter: {} and Running Loss: {}".format(i+1, running_loss))
        loss_list.append(running_loss)
        running_loss = 0

        # plot the reconstructed and original to see how they look
        fig, ax = plt.subplots()
        ax.plot(random_sequence.numpy())
        reconstructed = list(reconstructed[0].flatten())
        reconstructed.reverse()
        reconstructed = np.array(reconstructed).flatten()
        ax.plot(reconstructed)
        plt.savefig(os.path.join(file_dir, f"reconstructed/{i+1}.pdf"))
        plt.close()

# and observe loss dropping...
plt.plot(loss_list)
plt.savefig("loss.pdf")