import torch
from torch import nn
import numpy as np


def to_device(tensor, device):
    if device == "cpu":
        return tensor
    else:
        return tensor.cuda(device)


class SeqEncoder(nn.Module):
    """
    SeqEncoder takes a sequence and learns an embedding using a recurrent neural network
    """
    def __init__(self, input_dim, hidden_size, n_layers):
        super(SeqEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru_layer = nn.GRU(self.input_dim, self.hidden_size, self.n_layers, batch_first=True)

    def forward(self, input, lengths, hidden):
        input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
        output, hidden = self.gru_layer(input, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return output, hidden

    def initHidden(self, batch_size, device):
        return to_device(torch.zeros(self.n_layers, batch_size, self.hidden_size), device)


class SeqDecoder(nn.Module):
    """
    SeqDecoder takes the context vector (hidden state from the SeqEncoder and an initial seed to reproduce the
    entire sequence back)
    """
    def __init__(self, hidden_size, output_dim, n_layers):
        super(SeqDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.gru_layer = nn.GRU(self.output_dim, self.hidden_size, self.n_layers, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, input, hidden):
        output, hidden = self.gru_layer(input, hidden)
        output = self.output_layer(output)
        return output, hidden

    def initHidden(self, device):
        return to_device(torch.zeros(1, self.n_layers, self.hidden_size), device)

    def getInitialSeed(self, device):
        return to_device(torch.zeros(1, 1, self.output_dim), device)


class NoveltyDetectionModule:
    def __init__(self, input_dim, hidden_size, n_layers, device, lr, reg, epochs):

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        # instantiate the network objects
        self.encoder = SeqEncoder(self.input_dim, self.hidden_size, self.n_layers)
        self.decoder = SeqDecoder(self.hidden_size, self.input_dim, self.n_layers)

        # send the networks to devices
        self.encoder = to_device(self.encoder, self.device)
        self.decoder = to_device(self.decoder, self.device)

        # all the optimizers
        self.encoder_optimizer = torch.optim.Adagrad(self.encoder.parameters(), lr=lr, weight_decay=self.reg)
        self.decoder_optimier = torch.optim.Adagrad(self.decoder.parameters(), lr=lr, weight_decay=self.reg)

        # loss function
        self.loss_fn = nn.MSELoss()

        # archive set for the current step
        self.archive_set = []

    def add_behaviors(self, beh):
        # simply add the behaviors
        self.archive_set.append(beh)

    def fit_model(self, sequences):
        """
        :param sequence: sequence
        :return: None

        This function fits the given sequence to the model

        """

        # here the sequences are added to the archive set first
        for sequence in sequences:
            self.add_behaviors(sequence)

    def _pad_seq(self, seq, max_length):
        if max_length - seq.shape[0] > 0:
            seq_to_pad = torch.zeros(max_length - seq.shape[0], seq.shape[1]).double()
            seq_padded = torch.cat([seq, seq_to_pad])
            return seq_padded
        else:
            return seq

    def make_sequences_same_length(self):
        # Sort by length (descending)
        sequences = sorted(self.archive_set, key=lambda s: s.shape[0], reverse=True)

        # Pad sequences with zeros
        lengths = [s.shape[0] for s in sequences]
        sequences_padded = [self._pad_seq(s, max(lengths)) for s in sequences]

        # to tensor
        sequences_padded = torch.stack(sequences_padded)
        sequences_padded = to_device(sequences_padded, self.device)

        return sequences_padded, lengths

    def step(self):

        for i in range(self.epochs):

            # reset optimizers
            self.encoder_optimizer.zero_grad()
            self.decoder_optimier.zero_grad()

            # here, let's do a batch-forward
            sequences_padded, lengths = self.make_sequences_same_length()
            sequences_hidden = self.encoder.initHidden(sequences_padded.shape[0], self.device)
            encoder_outputs, encoder_hiddens = self.encoder.forward(sequences_padded.float(), lengths, sequences_hidden)

            loss_batch = 0
            reconstructed_sequences = []
            for idx in range(len(self.archive_set)):

                sequence = self.archive_set[idx]

                # get the length of the sequence
                seq_length = sequence.shape[0]

                # may have to re-shape the sequence here
                sequence = sequence.type(torch.float32)
                sequence = sequence.view(1, seq_length, self.input_dim)

                # send to device
                sequence = to_device(sequence, self.device)

                # initialize the inputs for decoder input
                decoder_hidden = encoder_hiddens[:, idx, :]
                decoder_hidden = decoder_hidden.view(self.decoder.n_layers, 1, -1)
                decoder_output = self.decoder.getInitialSeed(self.device)

                # now, reconstruct the sequence back in reverse order
                loss = 0
                reconstructed = []
                for i in range(seq_length):
                    decoder_output, decoder_hidden = self.decoder.forward(decoder_output, decoder_hidden.contiguous())
                    reconstructed.append(decoder_output.data.cpu().numpy())
                    loss += self.loss_fn(decoder_output, sequence[0, seq_length - 1 - i].reshape(decoder_output.shape))
                reconstructed_sequences.append(reconstructed)

                # average over sequences
                loss = loss / seq_length

                # add it to the loss batch
                loss_batch += loss

            # average it
            if len(self.archive_set) > 0:
                loss_batch = loss_batch / len(self.archive_set)

            # back prop
            loss_batch.backward()
            self.encoder_optimizer.step()
            self.decoder_optimier.step()

        # after training, empty the archive set
        self.archive_set = []

        return loss_batch.data.cpu().numpy(), reconstructed_sequences

    def get_novelty(self, sequence):
        """
        :param sequence: sequence
        :return: novelty scores

        This function applies Seq Encoder and Decoder approach and gets the loss as novelty score

        """

        # get the length of the sequence
        seq_length = sequence.shape[0]

        # re-shape the sequence
        sequence = sequence.view(1, seq_length, self.input_dim).type(torch.float32)

        # send sequence to the device
        sequence = to_device(sequence, self.device)

        # initialize the hidden state of the encoder network
        encoder_hidden = self.encoder.initHidden(1, self.device)

        # feed it in into the encoder network
        encoder_output, encoder_hidden = self.encoder.forward(sequence, [seq_length], encoder_hidden)

        # initialize the inputs for decoder input
        decoder_hidden = encoder_hidden
        decoder_output = self.decoder.getInitialSeed(self.device)

        # now, reconstruct the sequence back in reverse order
        loss = 0
        for i in range(seq_length):
            decoder_output, decoder_hidden = self.decoder.forward(decoder_output, decoder_hidden)
            loss += self.loss_fn(decoder_output, sequence[0, seq_length - 1 - i].reshape(decoder_output.shape))

        # average the loss
        loss = loss / seq_length

        # compute novelty as prediction error
        novelty = loss.data.cpu().numpy()

        return novelty
