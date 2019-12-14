import numpy as np
from archived.components.seq2seq_components import NoveltyDetectionModule
import torch
from matplotlib import pyplot as plt

# generate a time series (Mackey-Glass)
data = np.loadtxt("MackeyGlass_t17.txt")

# device
device = "cuda:0"

# novelty detection module (auto-encoder basically here)
autoencoder = NoveltyDetectionModule(input_dim=1, hidden_size=20, n_layers=2, device=device, lr=1e-1, reg=0.0, epochs=1)

# run for many iterations
loss_list = []
running_loss = 0
for i in range(1000):

    print(f"Running {i+1}")

    sequences = []

    for j in range(50):

        # at each iteration, sample a sub-sequence of variable length
        length = np.random.randint(30,100)
        index = np.random.randint(0, data.shape[0] - length)
        sequence = data[index:index+length]

        # to torch tensor
        sequence = torch.from_numpy(sequence).view((-1,1))
        sequences.append(sequence)

    # train it now
    autoencoder.fit_model(sequences)
    loss, reconstructed = autoencoder.step()

    running_loss += loss

    if (i+1) % 5 == 0:
        print("Iter: {} and Running Loss: {}".format(i+1, running_loss))
        loss_list.append(running_loss)
        running_loss = 0

        # plot the reconstructed and original to see how they look
        fig, ax = plt.subplots()
        ran_ix = np.random.randint(0, len(sequences))
        ax.plot(sequences[ran_ix].numpy())
        reconstructed = reconstructed[ran_ix]
        reconstructed.reverse()
        reconstructed = np.array(reconstructed).flatten()
        ax.plot(reconstructed)
        plt.show()