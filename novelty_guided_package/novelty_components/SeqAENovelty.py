from novelty_guided_package.novelty_components.abstract_detection_module import AbstractNoveltyDetector
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from novelty_guided_package.core_components.utility import to_device
import torch
from typing import List

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
    def __init__(self, n_input, n_hidden, n_layers, lr, sparsity_level, weight_decay, device):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_input = n_input
        self.lr = lr
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

    def _prepare_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence = sequence.type(torch.float32)
        sequence = sequence.view(1, sequence.shape[0], self.n_input)
        return sequence

    def forward(self, sequence: torch.Tensor):
        
        # prepare the sequence
        sequence = self._prepare_sequence(sequence)

        # get the length of the sequence
        seq_length = sequence.shape[1]

        # initialize the hidden state of the encoder network
        encoder_hidden = self.encoder.init_hidden(batch_size=1)

        # feed it in into the encoder network
        encoder_output, encoder_hidden = self.encoder.forward(sequence, encoder_hidden)

        # initialize the inputs for decoder input
        decoder_hidden = encoder_hidden
        decoder_output = sequence[0, seq_length-1].reshape(1, 1, -1)

        loss = 0
        reconstructed = []
        for i in range(seq_length-1):
            decoder_output, decoder_hidden = self.decoder.forward(decoder_output, decoder_hidden)
            reconstructed.append(decoder_output.data.numpy())
            loss += self.loss_fn(decoder_output, sequence[0, seq_length -2- i].reshape(decoder_output.shape))

        # average over sequences
        loss = loss / seq_length

        return reconstructed, loss

    def train(self, inputs: List[torch.tensor]):
        n_samples = len(inputs)
        loss_batch = 0
        for sequence in inputs:

            # reset optimizers
            self.encoder_optimizer.zero_grad()
            self.decoder_optimier.zero_grad()

            # forward the sequence
            reconstructed, loss = self.forward(sequence)
            
            # add it to the loss batch
            loss_batch += loss

        # average it
        if n_samples > 0:
            loss_batch = loss_batch / n_samples

        # back prop
        loss_batch.backward()
        self.encoder_optimizer.step()
        self.decoder_optimier.step()

        return loss_batch.data.numpy()


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
