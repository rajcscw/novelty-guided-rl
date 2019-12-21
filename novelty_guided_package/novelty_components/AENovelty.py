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

    def forward(self, input):
        encoded = self.encoder(input)

        # apply the sparseness
        sorted_indices = torch.argsort(encoded)
        top_indices = sorted_indices[:,:self.k_sparse]
        masks = torch.zeros_like(encoded)
        masks[:,top_indices] = 1.0
        encoded = encoded * masks
        output = self.decoder(encoded)
        return output

    # def forward(self, input):
    #     encoded = self.encoder(input)
    #     output = self.decoder(encoded)
    #     return output

    def train(self, inputs):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.zero_grad()
        outputs = self.forward(inputs)
        loss = AE.reconstruction_loss(inputs, outputs)
        loss.backward()
        optimizer.step()
        return loss.detach().cpu().numpy()

    @staticmethod
    def reconstruction_loss(input, predicted):
        criterion = torch.nn.MSELoss()(input, predicted).sum()
        return criterion


class AutoEncoderBasedDetection(AbstractNoveltyDetector):
    def __init__(self, n_input, n_hidden, lr, batch_size, device, sparsity_level, n_epochs=5):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.sparsity_level = sparsity_level
        self.n_epochs = n_epochs

        # model
        self.behavior_model = AE(n_input, n_hidden, lr, batch_size, sparsity_level)

        # set of behaviors (archive set)
        self.behaviors = []

    def _add_behaviors(self, beh):
        self.behaviors.append(beh)
        self.behaviors = self.behaviors[:100]

    def save_behaviors(self, behaviors):
        # add them to the behavior list
        for beh in behaviors:
            self._add_behaviors(beh.flatten())

    def step(self):
        # we fit the auto-encoder here
        behaviors = to_device(torch.Tensor(self.behaviors), self.device)
        for i in range(self.n_epochs):
            loss = self.behavior_model.train(behaviors)
            print(f"Total loss: {loss}")

    def get_novelty(self, behavior):
        behavior = behavior.reshape(1, -1)
        behavior = to_device(torch.Tensor(behavior), self.device)
        predicted = self.behavior_model.forward(behavior)
        novelty = float(self.behavior_model.reconstruction_loss(behavior, predicted).data.cpu().numpy())
        return novelty

    @classmethod
    def from_dict(cls, dict_config):
        return cls(**dict_config)
