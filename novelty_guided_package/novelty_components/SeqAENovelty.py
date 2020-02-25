from novelty_guided_package.novelty_components.abstract_detection_module import AbstractNoveltyDetector
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from novelty_guided_package.core_components.utility import to_device
import torch
from typing import List, Tuple
import numpy as np

class SeqEncoder(nn.Module):
    """
    SeqEncoder takes a sequence and learns an embedding using a recurrent neural network
    """
    def __init__(self, n_input: int, n_hidden: int, n_layers: int, sparsity_level: float, device: str):
        super(SeqEncoder, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.sparsity_level = sparsity_level
        self.device = device

        self.gru_layer = nn.GRU(self.n_input, self.n_hidden, self.n_layers, batch_first=True)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Forwards the input using the given hidden
        
        Arguments:
            input {torch.Tensor} -- input tensor with shape (batch_size x seq_len x n_input)
            hidden {torch.Tensor} -- hidden tensor with shape (n_layers x batch_size x n_input)
        
        Returns:
            torch.Tensor -- returns output and hidden representation of RNN
        """
        output, hidden = self.gru_layer(input, hidden)
        hidden = self._apply_sparsity(hidden)
        return output, hidden

    def _apply_sparsity(self, encoded: torch.Tensor) -> torch.Tensor:
        """Applies sparsity to the inner layer of hidden representation
        
        Arguments:
            encoded {torch.Tensor} -- hidden vector of RNN with shape (n_layers x batch_size x n_input)

        Returns:
            torch.Tensor -- returns output and hidden representation of RNN
        """
        inner_most_layer = encoded[self.n_layers-1]
        sorted_indices = torch.argsort(inner_most_layer, descending=True)
        k_sparse = int(inner_most_layer.shape[1] * self.sparsity_level)
        non_top_indices = sorted_indices[:,k_sparse:]
        masks = torch.ones_like(encoded)
        masks[self.n_layers-1, :, non_top_indices] = 0.0
        encoded = encoded * masks
        return encoded

    def init_hidden(self, batch_size):
        return to_device(torch.zeros(self.n_layers, batch_size, self.n_hidden), self.device)


class SeqDecoder(nn.Module):
    """
    SeqDecoder takes the context vector (hidden state from the SeqEncoder and an initial seed to reproduce the
    entire sequence back)
    """
    def __init__(self, n_hidden: int, n_output: int, n_layers: int, device: str):
        super(SeqDecoder, self).__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.device = device

        self.gru_layer = nn.GRU(self.n_output, self.n_hidden, self.n_layers, batch_first=True)
        self.output_layer = nn.Linear(self.n_hidden, self.n_output)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        output, hidden = self.gru_layer(input, hidden)
        output = self.output_layer(output)
        return output, hidden

class SeqAE(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, lr, batch_size, sparsity_level, weight_decay, device):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_input = n_input
        self.lr = lr
        self.batch_size = batch_size
        self.sparsity_level = sparsity_level
        self.weight_decay = weight_decay
        self.device = device
        
        # set up encoder and decoder modules
        self.encoder = SeqEncoder(n_input, n_hidden, n_layers, sparsity_level, device)
        self.decoder = SeqDecoder(n_hidden, n_input, n_layers, device)
        
        # optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=self.weight_decay)
        self.decoder_optimier = torch.optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=self.weight_decay)

        # loss function
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def _pad_sequence(sequence: torch.Tensor, to_length: int):
        to_pad_len = to_length - sequence.shape[0]
        if to_pad_len > 0:
            pad = torch.zeros((to_length - sequence.shape[0], sequence.shape[1]))
            padded_sequence = torch.cat([sequence, pad], dim=0)
            padded_sequence = padded_sequence.reshape((1, to_length, -1))
            return padded_sequence
        else:
            padded_sequence = sequence.reshape((1, to_length, -1))
            return padded_sequence

    def _prepare_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        # pad the sequences to be of the same length
        original_seq_lengths = [sequence.shape[0] for sequence in sequences]
        max_length = max(original_seq_lengths)
        sequences = [SeqAE._pad_sequence(sequence.type(torch.float32), max_length) for sequence in sequences]
        sequences = torch.cat(sequences, dim=0)
        return sequences, original_seq_lengths

    def forward(self, sequences: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor, List[np.ndarray]]:
        """Forward operation
        
        Arguments:
            sequences {List[torch.Tensor]} -- a list of sequences each in the shape [seq_len x dim]
        
        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, List[np.ndarray]] -- returns a tuple of reconstructed sequences, loss_batch, losses 
        """
        
        # prepare the sequences
        sequences, original_seq_lengths = self._prepare_sequences(sequences)
        max_length = max(original_seq_lengths)

        # encode
        encoder_hidden = self.encoder.init_hidden(batch_size=sequences.shape[0])
        encoder_output, encoder_hidden = self.encoder.forward(sequences, encoder_hidden)

        # decode
        decoder_hidden = encoder_hidden
        decoder_output = torch.zeros((sequences.shape[0], 1, self.n_input))

        loss_batch = 0
        current_batch_size = decoder_output.shape[0]
        reconstructed_sequences = torch.zeros((current_batch_size, max_length, self.n_input))
        original_sequences = torch.zeros((current_batch_size, max_length, self.n_input))
        for i in range(max_length):
            t = max_length - 1 - i
            decoder_output, decoder_hidden = self.decoder.forward(decoder_output, decoder_hidden)
            reconstructed_sequences[:,i,:] = decoder_output[:, 0, :]
            original_sequences[:,i,:] = sequences[:,t, :] # in reversed order

        # compute loss for sequences
        loss_batch = 0
        reconstructed_sequences_as_list = []
        loss_per_sequences = []
        for i in range(current_batch_size):
            # for considering variable length sequences
            # now, we consider only clipped versions of it
            seq_length = original_seq_lengths[i]

            # the original sequence which is now in reversed order, the first few elements could be the padded ones 
            # so the last seq length elements are chosen
            original = original_sequences[i, :, :][max_length-seq_length:]

            # reconstructed must consider only the first seq length elements
            reconstructed = reconstructed_sequences[i, :, :][:seq_length]
            reconstructed_sequences_as_list.append(reconstructed.detach().numpy().reshape((-1, self.n_input))) 
            loss = self.loss_fn(reconstructed, original) / seq_length # divide by seq length
            loss_per_sequences.append(loss.detach().numpy())
            loss_batch += loss
        loss_batch = loss_batch / current_batch_size # divide by batch size

        return reconstructed_sequences_as_list, loss_batch, loss_per_sequences

    def train(self, inputs: List[torch.tensor]):
        n_samples = len(inputs)
        current_batch_idx = 0
        total_loss = 0
        while current_batch_idx < n_samples:
            # reset optimizers
            self.encoder_optimizer.zero_grad()
            self.decoder_optimier.zero_grad()

            # forward the sequence
            sequences = inputs[current_batch_idx:current_batch_idx+self.batch_size]
            reconstructed, loss, _ = self.forward(sequences)
            
            # add it to the loss batch
            total_loss += loss

            # back prop
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimier.step()

            # next batch
            current_batch_idx += self.batch_size

        return total_loss.data.numpy()

class SequentialAutoEncoderBasedDetection(AbstractNoveltyDetector):
    def __init__(self, n_input, n_hidden, n_layers, lr, device, sparsity_level, archive_size, n_epochs):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        self.device = device
        self.sparsity_level = sparsity_level
        self.n_epochs = n_epochs
        self.archive_size = archive_size

        # model
        self.behavior_model = SeqAE(n_input, n_hidden, n_layers, lr, sparsity_level, device)

        # set of behaviors (archive set)
        self.behaviors = []

    def _add_behaviors(self, beh):
        self.behaviors.append(beh)
        self.behaviors = self.behaviors[:self.archive_size]

    def save_behaviors(self, behaviors):
        # add them to the behavior list
        for beh in behaviors:
            self._add_behaviors(beh.flatten())

    def step(self):
        # we fit the auto-encoder here
        behaviors = to_device(torch.Tensor(self.behaviors), self.device)
        for i in range(self.n_epochs):
            loss = self.behavior_model.train(behaviors)

    def get_novelty(self, behavior):
        with torch.no_grad():
            behavior = behavior.reshape(1, -1)
            behavior = to_device(torch.Tensor(behavior), self.device)
            predicted = self.behavior_model.forward(behavior)
            novelty = float(self.behavior_model.reconstruction_loss(behavior, predicted).data.cpu().numpy())
            return novelty

    @classmethod
    def from_dict(cls, dict_config):
        return cls(**dict_config)
