import numpy as np
from novelty_guided_package.novelty_components.SeqAENovelty import SeqAE
import torch
from matplotlib import pyplot as plt
import os

# generate a time series (Mackey-Glass)
file_dir = os.path.dirname(os.path.realpath(__file__))
data = np.loadtxt(os.path.join(file_dir, "MackeyGlass_t17.txt"))

# novelty detection module (auto-encoder basically here)
autoencoder = SeqAE(n_input=1, n_hidden=50, n_layers=2, lr=1e-2, sparsity_level=0.7, weight_decay=0, device="cpu")

# run for many iterations
loss_list = []
running_loss = 0
for i in range(10000):
    # at each iteration, sample a sub-sequence of variable length
    length = np.random.randint(70,100)
    index = np.random.randint(0, data.shape[0] - length)
    sequence = data[index:index+length]

    # to torch tensor
    sequence = torch.from_numpy(sequence).reshape(-1,1)

    # train it now
    loss = autoencoder.train([sequence])
    running_loss += loss

    # reconstrcuted
    reconstructed, _ = autoencoder.forward(sequence)

    if (i+1) % 200 == 0:
        print("Iter: {} and Running Loss: {}".format(i+1, running_loss))
        loss_list.append(running_loss)
        running_loss = 0

        # plot the reconstructed and original to see how they look
        fig, ax = plt.subplots()
        ax.plot(sequence.numpy())
        reconstructed.reverse()
        reconstructed = np.array(reconstructed).flatten()
        ax.plot(reconstructed)
        plt.savefig(os.path.join(file_dir, f"reconstructed/{i+1}.pdf"))

# and observe loss dropping...
plt.plot(loss_list)
plt.savefig("loss.pdf")