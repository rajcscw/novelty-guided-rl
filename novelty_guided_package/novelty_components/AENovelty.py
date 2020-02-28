from novelty_guided_package.novelty_components.abstract_detection_module import AbstractNoveltyDetector
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from novelty_guided_package.core_components.utility import to_device
import torch


class AE(nn.Module):
    def __init__(self, n_input, n_hidden, lr, batch_size, sparsity_level):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.lr = lr
        self.sparsity_level = sparsity_level
        self.batch_size = batch_size
        self.encoder_layers, self.decoder_layers = self._get_layers(n_hidden)
        self.loss_criterion = torch.nn.MSELoss()

    def _get_layers(self, n_hidden: str):
        # separate the layering information
        n_hidden = n_hidden.split(",")
        n_hidden = [int(hidden) for hidden in n_hidden]

        encoder_layers = nn.ModuleList()
        decoder_layers = nn.ModuleList()
        
        input_size = self.n_input
        for output_size in n_hidden:
            encoder_layers.append(nn.Linear(input_size, output_size))
            decoder_layers.append(nn.Linear(output_size, input_size)) 
            input_size = output_size
        
        return encoder_layers, decoder_layers

    def get_index_array(self, indices: torch.Tensor):
        ix_indices_1 = []
        ix_indices_2 = []
        for ix_1 in range(indices.shape[0]):
            for ix_2 in indices[ix_1]:
                ix_indices_1.append(ix_1)
                ix_indices_2.append(int(ix_2))
        return torch.tensor(ix_indices_1), torch.tensor(ix_indices_2)

    def forward(self, input):
        # encoding
        encoded = input
        for layer in self.encoder_layers:
            encoded = torch.sigmoid(layer(encoded))
        
        # apply the sparseness
        sorted_indices = torch.argsort(encoded, descending=True)
        k_sparse = int(encoded.shape[1] * self.sparsity_level)
        top_indices = sorted_indices[:,:k_sparse]
        top_indices = self.get_index_array(top_indices)
        masks = torch.zeros_like(encoded)
        masks[top_indices] = 1.0
        encoded = encoded * masks

        # decoding
        decoded = encoded
        for i, layer in enumerate(reversed(self.decoder_layers)):
            if (i + 1) != len(self.decoder_layers):
                decoded = torch.sigmoid(layer(decoded))
            else:
                decoded = layer(decoded)

        return decoded 

    def train(self, inputs):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        current_batch_idx = 0
        total_loss = 0
        while current_batch_idx < inputs.shape[0]:
            self.zero_grad()
            batch_inputs = inputs[current_batch_idx:current_batch_idx+self.batch_size]
            outputs = self.forward(batch_inputs)
            loss = AE.reconstruction_loss(batch_inputs, outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss
            current_batch_idx += self.batch_size
        return total_loss.detach().cpu().numpy() / inputs.shape[0]

    @staticmethod
    def reconstruction_loss(input, predicted):
        criterion = torch.nn.MSELoss()(input, predicted).sum()
        return criterion


class AutoEncoderBasedDetection(AbstractNoveltyDetector):
    def __init__(self, n_input, n_hidden, lr, batch_size, device, sparsity_level, archive_size, n_epochs):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.sparsity_level = sparsity_level
        self.n_epochs = n_epochs
        self.archive_size = archive_size

        # model
        self.behavior_model = AE(n_input, n_hidden, lr, batch_size, sparsity_level)

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
