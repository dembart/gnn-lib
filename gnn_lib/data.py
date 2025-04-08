import pickle
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset
from vesin import NeighborList
import lmdb


class Data(object):
    
    __slots__ = [
        'node_attr', 'edge_index', 'edge_attr', 'edge_vector', 
        'edge_length', 'offset', 'positions', 'energy', 
        'numbers', 'forces', 'stress', 'batch'
    ]
    
    def __init__(
        self,
        node_attr=None,
        numbers=None,
        edge_index=None,
        edge_attr=None,
        edge_vector=None,
        edge_length=None,
        offset=None,
        positions=None,
        energy=None,
        forces=None, 
        stress=None,
    ):
        self.node_attr = node_attr
        self.numbers = numbers
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_vector = edge_vector
        self.edge_length = edge_length 
        self.offset = offset
        self.positions = positions
        self.energy = energy
        self.forces = forces
        self.stress = stress
        self.batch = None

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.node_attr.size(0)
        return 0

    def __cat_dim__(self, key, value):
        if key in ['node_attr', 'edge_attr', 'edge_vector', 
                   'edge_length', 'positions', 'energy', 'stress',
                    'numbers', 'forces']:
            return 0  
        return None

    def to(self, device):
        for key in self.__slots__:
            if key == 'batch':  
                continue
            attr = getattr(self, key)
            if attr is not None:
                setattr(self, key, attr.to(device, non_blocking=True))
        if self.batch is not None:
            self.batch = self.batch.to(device, non_blocking=True)
        return self
    


def atoms2data(atoms,
               r_cut=None,
               energy_key=None,
               forces_key=None,
               stress_key=None,
               n_max_neighbors=None,
               node_featurizer=None,
               edge_featurizer=None,
               ):
    
    calculator = NeighborList(cutoff=r_cut, full_list=True)
    i, j, ij_length, offset, ij_vector = calculator.compute(
        points=atoms.positions,
        box=atoms.cell,
        periodic=True in atoms.pbc, 
        quantities="ijdSD"
    )

    if n_max_neighbors is not None:
        filtered_indices = []
        for atom_idx in np.unique(i):
            mask = (i == atom_idx)
            indices = np.where(mask)[0]
            sorted_indices = indices[np.argsort(ij_length[mask])]
            filtered_indices.extend(sorted_indices[:n_max_neighbors])
        i = i[filtered_indices]
        j = j[filtered_indices]
        ij_length = ij_length[filtered_indices]
        offset = offset[filtered_indices]
        ij_vector = ij_vector[filtered_indices]

    ij = torch.stack([torch.LongTensor(i), torch.LongTensor(j)])
    data = Data(
        edge_index=ij,
        edge_vector=torch.tensor(ij_vector, dtype=torch.float32),
        edge_length=torch.tensor(ij_length, dtype=torch.float32),
        positions=torch.tensor(atoms.positions, dtype=torch.float32),
        offset=torch.tensor(offset, dtype=torch.float32),
        numbers=torch.LongTensor(atoms.get_atomic_numbers()),
        energy=None if energy_key is None else torch.tensor(atoms.info[energy_key], dtype=torch.float32),
        forces=None if forces_key is None else torch.tensor(atoms.info[forces_key], dtype=torch.float32),
        stress=None if stress_key is None else torch.tensor(atoms.info[stress_key], dtype=torch.float32),
    )
    if node_featurizer is not None:
        data.node_attr = node_featurizer(data)
    if edge_featurizer is not None:
        data.edge_attr = edge_featurizer(data)
    return data



class InMemoryDataset(Dataset):

    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self._preprocessed = False


    @classmethod
    def from_file(cls, path):
        data_list = torch.load(path)
        return cls(data_list)
    
    @classmethod
    def from_data_list(cls, data_list):
        return cls(data_list)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform:
            data = self.transform(data)
        return data
    
    def preprocess(self):
        if not self._preprocessed:
            self.data_list = [self._process_data(d) for d in self.data_list]
            self._preprocessed = True
            
    @staticmethod
    def _process_data(data):
        processed = Data()
        
        index_keys = ['edge_index', 'numbers']
        float_keys = ['node_attr', 'edge_attr', 'edge_vector', 
                    'edge_length', 'positions', 'energy',
                    'forces', 'stress']

        for key in index_keys:
            attr = getattr(data, key)
            if attr is not None:
                if not isinstance(attr, torch.Tensor):
                    processed_attr = torch.tensor(attr, dtype=torch.long)
                else:
                    processed_attr = attr.to(torch.long)
                setattr(processed, key, processed_attr)

        for key in float_keys:
            attr = getattr(data, key)
            if attr is not None:
                if not isinstance(attr, torch.Tensor):
                    processed_attr = torch.tensor(attr, dtype=torch.float32)
                else:
                    processed_attr = attr.float()
                setattr(processed, key, processed_attr)
        return processed



class LMDBDataset(Dataset):

    def __init__(self, lmdb_path, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env = None
        self.num_graphs = 0
        self._open_env()
        
    def _open_env(self):
        self.env = lmdb.open(self.lmdb_path,
                            readonly=True,
                            max_readers=64,
                            lock=False,
                            readahead=False,
                            meminit=False,
                            max_spare_txns=64
                            )
        
        with self.env.begin() as txn:
            self.num_graphs = pickle.loads(txn.get(b'num_graphs'))

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        if self.env is None:
            self._open_env()
            
        with self.env.begin() as txn:
            data = torch.load(BytesIO(txn.get(f'graph_{idx}'.encode())))
            
        if self.transform:
            data = self.transform(data)
        return data

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __del__(self):
        self.close()



DEFAULT_MAPPER = {
                    'H': 0,
                    'He': 1,
                    'Li': 2,
                    'Be': 3,
                    'B': 4,
                    'C': 5,
                    'N': 6,
                    'O': 7,
                    'F': 8,
                    'Ne': 9,
                    'Na': 10,
                    'Mg': 11,
                    'Al': 12,
                    'Si': 13,
                    'P': 14,
                    'S': 15,
                    'Cl': 16,
                    'Ar': 17,
                    'K': 18,
                    'Ca': 19,
                    'Sc': 20,
                    'Ti': 21,
                    'V': 22,
                    'Cr': 23,
                    'Mn': 24,
                    'Fe': 25,
                    'Co': 26,
                    'Ni': 27,
                    'Cu': 28,
                    'Zn': 29,
                    'Ga': 30,
                    'Ge': 31,
                    'As': 32,
                    'Se': 33,
                    'Br': 34,
                    'Kr': 35,
                    'Rb': 36,
                    'Sr': 37,
                    'Y': 38,
                    'Zr': 39,
                    'Nb': 40,
                    'Mo': 41,
                    'Tc': 42,
                    'Ru': 43,
                    'Rh': 44,
                    'Pd': 45,
                    'Ag': 46,
                    'Cd': 47,
                    'In': 48,
                    'Sn': 49,
                    'Sb': 50,
                    'Te': 51,
                    'I': 52,
                    'Xe': 53,
                    'Cs': 54,
                    'Ba': 55,
                    'La': 56,
                    'Ce': 57,
                    'Pr': 58,
                    'Nd': 59,
                    'Pm': 60,
                    'Sm': 61,
                    'Eu': 62,
                    'Gd': 63,
                    'Tb': 64,
                    'Dy': 65,
                    'Ho': 66,
                    'Er': 67,
                    'Tm': 68,
                    'Yb': 69,
                    'Lu': 70,
                    'Hf': 71,
                    'Ta': 72,
                    'W': 73,
                    'Re': 74,
                    'Os': 75,
                    'Ir': 76,
                    'Pt': 77,
                    'Au': 78,
                    'Hg': 79,
                    'Tl': 80,
                    'Pb': 81,
                    'Bi': 82,
                    'Po': 83,
                    'At': 84,
                    'Rn': 85,
                    'Fr': 86,
                    'Ra': 87,
                    'Ac': 88,
                    'Th': 89,
                    'Pa': 90,
                    'U': 91,
                    'Np': 92,
                    'Pu': 93
}
