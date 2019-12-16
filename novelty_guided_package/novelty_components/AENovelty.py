from novelty_guided_package.novelty_components.abstract_detection_module import AbstractNoveltyDetector
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from novelty_guided_package.core_components.utility import to_device
import torch


class AE(nn.Module):
    def __init__(self, n_input, n_hidden, lr, batch_size):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.lr = lr
        self.batch_size = batch_size
        self.encoder = nn.Linear(n_input, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_input)

    def forward(self, input):
        encoded = self.encoder(input)
        output = self.decoder(encoded)
        return output

    def train(self, inputs):
        optimizer = optim.SGD(self.parameters(), lr=self.lr)
        data_set = TensorDataset((inputs))
        data_loader = DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
        for batch in data_loader:
            input = batch[0]
            optimizer.zero_grad()
            output = self.forward(input)
            loss = self.reconstruction_loss(input, output)
            loss.backward()
            optimizer.step()

    @staticmethod
    def reconstruction_loss(input, predicted):
        criterion = torch.nn.MSELoss()(input, predicted).sum()
        return criterion


class AutoEncoderBasedDetection(AbstractNoveltyDetector):
    def __init__(self, n_input, n_hidden, lr, batch_size, device, n_epochs=1):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.n_epochs = n_epochs

        # model
        self.behavior_model = AE(n_input, n_hidden, lr, batch_size)

        # set of behaviors (archive set)
        self.behaviors = []

    def _add_behaviors(self, beh):
        self.behaviors.append(beh)

    def save_behaviors(self, behaviors):
        # add them to the behavior list
        for beh in behaviors:
            self._add_behaviors(beh.flatten())

    def step(self):
        # we fit the auto-encoder here
        self.behaviors = to_device(torch.Tensor(self.behaviors), self.device)
        for i in range(self.n_epochs):
            self.behavior_model.train(self.behaviors)

        # and finally, clear the archive set at every epoch
        self.behaviors = []

    def get_novelty(self, behavior):
        behavior = behavior.flatten()
        behavior = to_device(torch.Tensor(behavior), self.device)
        predicted = self.behavior_model.forward(behavior)
        novelty = float(self.behavior_model.reconstruction_loss(behavior, predicted).data.cpu().numpy())
        return novelty

    @classmethod
    def from_dict(cls, dict_config):
        return cls(**dict_config)
