import lmdb
import pickle
from io import BytesIO
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from vesin import NeighborList



class Data:
    def __init__(
        self,
        positions: Tensor | None = None,
        numbers: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_vector: Tensor | None = None,
        edge_length: Tensor | None = None,
        offset: Tensor | None = None,
        cell: Tensor | None = None,
        node_attr: Tensor | None = None,
        edge_attr: Tensor | None = None,
        energy: Tensor | None = None,
        forces: Tensor | None = None,
        stress: Tensor | None = None,
        num_atoms: int | None = None,
    ) -> None:
        self.positions: Tensor | None = positions
        self.numbers: Tensor | None = numbers
        self.node_attr: Tensor | None = node_attr
        self.edge_index: Tensor | None = edge_index
        self.edge_vector: Tensor | None = edge_vector
        self.edge_length: Tensor | None = edge_length
        self.edge_attr: Tensor | None = edge_attr
        self.offset: Tensor | None = offset
        self.cell: Tensor | None = cell
        self.energy: Tensor | None = energy
        self.forces: Tensor | None = forces
        self.stress: Tensor | None = stress
        self.num_atoms: Tensor | None = num_atoms
        self.batch: Tensor | None = None
        

    def to(self, device: torch.device) -> "Data":
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device, non_blocking=True))
        return self

    def __repr__(self) -> str:
        attrs = []
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                attrs.append(f"{key}: {tuple(value.shape)}")
            elif value is not None:
                attrs.append(f"{key}: {type(value).__name__}")
        return f"Data({', '.join(attrs)})"



def atoms2data(
    atoms,
    r_cut: float | None = None,
    energy_key: str | None = None,
    forces_key: str | None = None,
    stress_key: str | None = None,
    n_max_neighbors: int | None = None,
) -> Data:


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

    edge_index = torch.stack([torch.as_tensor(i, dtype=torch.long),
                              torch.as_tensor(j, dtype=torch.long)])

    data = Data(
        num_atoms=torch.tensor([len(atoms)], dtype=torch.long),
        numbers=torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long),
        positions=torch.as_tensor(atoms.positions, dtype=torch.float32),
        edge_index=edge_index,
        edge_length=torch.tensor(ij_length, dtype=torch.float32),
        offset=torch.as_tensor(offset, dtype=torch.float32),
        cell=None if not all(atoms.pbc) else torch.as_tensor(np.array(atoms.cell), dtype=torch.float32),
        energy=None if energy_key is None else torch.as_tensor(atoms.info[energy_key], dtype=torch.float32),
        forces=None if forces_key is None else torch.as_tensor(atoms.info[forces_key], dtype=torch.float32),
        stress=None if stress_key is None else torch.as_tensor(atoms.info[stress_key], dtype=torch.float32),
    )
    return data



class InMemoryDataset(Dataset):
    """In-memory dataset for Data objects."""

    def __init__(
        self,
        data_list: list[Data],
        transform=None,
    ):
        self.data_list = data_list
        self.transform = transform

    @classmethod
    def from_file(cls, path: str):
        data_list = torch.load(path, weights_only=False)
        return cls(data_list)
    

    @classmethod
    def from_data_list(cls, data_list: list[Data]) -> "InMemoryDataset":
        return cls(data_list)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data



class LMDBDataset(Dataset):
    """LMDB-backed dataset safe for multiprocessing DataLoader."""

    def __init__(
        self,
        lmdb_path: str,
        transform=None,
    ):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.env: lmdb.Environment | None = None
        self.num_graphs: int | None = None

    def _open_env(self) -> None:
        self.env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            max_readers=64,
            lock=False,
            readahead=False,
            meminit=False,
            max_spare_txns=64,
        )

        with self.env.begin() as txn:
            self.num_graphs = pickle.loads(txn.get(b"num_graphs"))

    def __len__(self) -> int:
        if self.env is None:
            self._open_env()
        return self.num_graphs  # type: ignore

    def __getitem__(self, idx: int) -> Data:
        if self.env is None:
            self._open_env()

        with self.env.begin() as txn:
            raw = txn.get(f"graph_{idx}".encode())
            data: Data = torch.load(BytesIO(raw))

        if self.transform is not None:
            data = self.transform(data)

        return data

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            self.num_graphs = None

    def __del__(self):
        self.close()