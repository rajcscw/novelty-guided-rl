from sklearn.neighbors import NearestNeighbors
import numpy as np
import warnings
from novelty_guided_package.novelty_components.abstract_detection_module import AbstractNoveltyDetector


class NearestNeighborDetection(AbstractNoveltyDetector):
    def __init__(self, n_neighbors, archive_size):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.archive_size = archive_size

        # model
        self.behavior_model = NearestNeighbors(n_neighbors=self.n_neighbors)

        # set of behaviors (archive set)
        self.behaviors = []

        # counters to set if the models are fitted
        self.fitted_count = 0

    def _add_behaviors(self, beh):
        self.behaviors.append(beh)
        self.behaviors = self.behaviors[-self.archive_size:]

    def save_behaviors(self, behaviors):

        # add them to the behavior list
        for beh in behaviors:
            self._add_behaviors(beh)
            self.fitted_count += 1

        # fit the model now
        data_points = np.array(self.behaviors).reshape(len(self.behaviors), -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.behavior_model.fit(data_points)

    def step(self):
        pass

    def get_novelty(self, behavior):
        # get nearest neighbors and its distance to the current policy
        if self.fitted_count > self.behavior_model.n_neighbors:
            data_point = behavior.reshape(1, -1)
            neighbors_distance, _ = self.behavior_model.kneighbors(data_point)
            novelty = np.sum(neighbors_distance) / self.behavior_model.n_neighbors
        else:
            novelty = 0.0
        return novelty

    @classmethod
    def from_dict(cls, dict_config):
        return  cls(**dict_config)
