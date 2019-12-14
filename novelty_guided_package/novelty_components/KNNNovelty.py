from sklearn.neighbors import NearestNeighbors
import numpy as np
import warnings


class NoveltyDetectionModule:
    def __init__(self, k, limit):
        # copy of params
        self.k = k
        self.behavior_limit = limit

        # model
        self.behavior_model = NearestNeighbors(n_neighbors=k)

        # set of behavoirs (archive set)
        self.behavoirs = []

        # counters to set if the models are fitted
        self.fitted_count = 0

    @classmethod
    def from_dict(cls, config):
        return cls(**config)

    def add_behaviors(self, beh):
        self.behavoirs.append(beh)
        self.behavoirs = self.behavoirs[-self.behavior_limit:]

    def fit_model(self, behaviors):

        # add them to the behavoir list
        for beh in behaviors:
            self.add_behaviors(beh)
            self.fitted_count += 1

        # fit the model now
        data_points = np.array(self.behavoirs).reshape(len(self.behavoirs), -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.behavior_model.fit(data_points)

    def step(self):
        pass

    def get_novelty(self, beh):
        # get nearest neighbors and its distance to the current policy
        if self.fitted_count > self.behavior_model.n_neighbors:
            data_point = beh.reshape(1, -1)
            neighbors_distance, _ = self.behavior_model.kneighbors(data_point)
            novelty = np.sum(neighbors_distance) / self.behavior_model.n_neighbors
        else:
            novelty = 0.0
        return novelty