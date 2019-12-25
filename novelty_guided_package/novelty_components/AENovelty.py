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
        self.k_sparse = int(sparsity_level * n_hidden)
        self.batch_size = batch_size
        self.encoder = nn.Linear(n_input, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_input)
        self.loss_criterion = torch.nn.MSELoss()

    def forward(self, input):
        encoded = torch.sigmoid(self.encoder(input))

        # apply the sparseness
        sorted_indices = torch.argsort(encoded)
        top_indices = sorted_indices[:,:self.k_sparse]
        masks = torch.zeros_like(encoded)
        masks[:,top_indices] = 1.0
        encoded = encoded * masks
        output = self.decoder(encoded)
        return output

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
