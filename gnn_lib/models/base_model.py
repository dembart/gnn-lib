from abc import ABC, abstractmethod
import torch
from torch import nn

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()


    @abstractmethod
    def forward(self, data):
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, atomic_types_mapper, config):
        """
        Buil model from config and atomic_types_mapper

        Parameters
        ----------

        atomic_types_mapper: dict
            mapper, atomic_symbol -> index

        config: dict (or dotdict)
            configuration file

        """
        return cls(atomic_types_mapper, config)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def from_checkpoint(self, path):
        self.load_state_dict(torch.load(path)['model_state'])

    def size(self):
        total_size = 0
        for p in self.parameters():
            size = 1
            for s in p.size():
                size *= s
            total_size += size
        return total_size