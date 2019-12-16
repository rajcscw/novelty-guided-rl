from abc import ABC, abstractmethod
from typing import Dict


class AbstractNoveltyDetector(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def save_behaviors(self, behaviors):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_novelty(self, behavior):
        pass

    @classmethod
    @abstractmethod
    def from_dict(self, dict_config: Dict):
        pass