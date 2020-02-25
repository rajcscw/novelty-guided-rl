import numpy as np
from novelty_guided_package.novelty_components.SeqAENovelty import SeqAE
import torch
from matplotlib import pyplot as plt
import os

# generate a time series (Mackey-Glass)
file_dir = os.path.dirname(os.path.realpath(__file__))
data = np.loadtxt(os.path.join(file_dir, "MackeyGlass_t17.txt"))

# novelty detection module (auto-encoder basically here)
batch_size = 50
autoencoder = SeqAE(n_input=1, n_hidden=50, n_layers=2, lr=1e-2, batch_size=50, sparsity_level=1.0, weight_decay=0, device="cpu")

# run for many iterations
loss_list = []
running_loss = 0
max_iters = 10000
for i in range(max_iters):
    # at each iteration, sample a sub-sequence of variable length
    length = np.random.randint(70,100)

    # generate sequences
    indices = np.random.randint(0, data.shape[0] - length, batch_size)
    sequences = [torch.from_numpy(data[index:index+length]).reshape(-1,1) for index in indices]

    # train it now
    loss = autoencoder.train(sequences)
    running_loss += loss

    # reconstrcuted
    random_idx = np.random.choice(range(batch_size))
    random_sequence = sequences[random_idx]
    reconstructed, _ = autoencoder.forward([random_sequence])

    if (i+1) % 100 == 0:
        print("Iter: {} and Running Loss: {}".format(i+1, running_loss))
        loss_list.append(running_loss)
        running_loss = 0

        # plot the reconstructed and original to see how they look
        fig, ax = plt.subplots()
        ax.plot(random_sequence.numpy())
        reconstructed.reverse()
        reconstructed = np.array(reconstructed).flatten()
        ax.plot(reconstructed)
        plt.savefig(os.path.join(file_dir, f"reconstructed/{i+1}.pdf"))

# and observe loss dropping...
plt.plot(loss_list)
plt.savefig("loss.pdf")