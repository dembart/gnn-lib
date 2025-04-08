import torch
from torch import nn
from ase.data import atomic_numbers

class OneHot(nn.Module):
    
    def __init__(self, atomic_types_mapper=None, **kwargs):
        super().__init__()
        
        self.mapper = {
            atomic_numbers[key]: value 
            for key, value in atomic_types_mapper.items()
        }
        
        self.reverse_mapper = {v: k for k, v in atomic_types_mapper.items()}
        self.num_classes = len(atomic_types_mapper)
        self.register_buffer("_eye_matrix", torch.eye(self.num_classes))

    def forward(self, data):

        class_indices = torch.tensor([
            self.mapper[int(num)] 
            for num in data.numbers
        ], dtype=torch.long, device=self._eye_matrix.device)
        
        unrecognized = set(data.numbers.cpu().numpy()) - set(self.mapper.keys())
        if unrecognized:
            raise ValueError(f"Unrecognized atomic numbers: {unrecognized}")
        return self._eye_matrix[class_indices]



class GaussianFilter(nn.Module):
    
    def __init__(self, r_min=1.0, r_max=6.0, num_edge_features=64):

        super(GaussianFilter, self).__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.num_edge_features = num_edge_features
        _filter = torch.linspace(r_min, r_max, num_edge_features)
        step = (r_max - r_min) / num_edge_features
        self.register_buffer("_filter", _filter)
        self.register_buffer("step", torch.tensor(step))

    def forward(self, data):

        if not isinstance(data.edge_length, torch.Tensor):
            edge_lengths = torch.tensor(data.edge_length, dtype=torch.float32)
        edge_lengths = edge_lengths.to(self._filter.device)
        diff = edge_lengths[:, None] - self._filter
        out = torch.exp(-0.5 * torch.square(diff / self.step))
        return out
    

