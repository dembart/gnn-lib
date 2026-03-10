import os
import json
import pickle
import lmdb
from io import BytesIO
from ase.io import iread
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from .graph_data import Data, LMDBDataset, InMemoryDataset, atoms2data#, DEFAULT_MAPPER



def build_dataloader(config, data_split):

    dataset = build_dataset(config, data_split)    
    if data_split == 'train':
        batch_size = config['training']['train_batch_size']
    else:
        batch_size = config['training']['val_batch_size']
    
    loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                shuffle=True if data_split == 'train' else False,
                num_workers=config['num_workers'],
                pin_memory=True,
                #persistent_workers=True if config['num_workers'] else False,  
            )
    return loader



def build_dataset(config, data_split):
    processed_data_path = config['data']['processed_data_path']
    if config['data']['use_lmdb']:
        if not os.path.exists(processed_data_path + f'/data_{data_split}'):
            data_generator = generate_dataset(config, data_split)
            os.makedirs(processed_data_path, exist_ok=True)
            dataset = create_lmdb_dataset(data_generator,
                                          processed_data_path, data_split=data_split)
        else:
            dataset = LMDBDataset(processed_data_path + f'/{data_split}')
    else:
        if not os.path.exists(processed_data_path + f'/data_{data_split}.pt'):
            print('check check', data_split)
            data_generator = generate_dataset(config, data_split)
            dataset = create_inmemory_dataset(data_generator,
                                                    folder = processed_data_path,
                                                    data_split = data_split
                                                    )
        else: 
            dataset = InMemoryDataset.from_file(processed_data_path + f'/data_{data_split}.pt')
    return dataset


def generate_dataset(config, data_split):

    if data_split not in ['train', 'val', 'test']:
        raise ValueError

    ref_energies = None
    if config.model.E0s:
        print('Will use E0s. The model will predict residual energy.')
        ref_energies = torch.zeros(config.model.in_dim)
        for z, e0 in config.model.E0s.items():
            ref_energies[z] = e0
        

    data_file = config.data[data_split]
    energy_key = None if config.data.energy_key is False else config.data.energy_key
    forces_key = None if config.data.forces_key is False else config.data.forces_key
    stress_key = None if config.data.stress_key is False else config.data.stress_key
            
    for atoms in iread(data_file):
        data = atoms2data(
            atoms,
            r_cut=config['data']['r_cut'],
            energy_key=energy_key,
            forces_key=forces_key,
            stress_key=stress_key,
            n_max_neighbors=config.data.n_max_neighbors if config.data.n_max_neighbors else None,
        )
    
        # atomic reference energies
        if ref_energies is not None:
            atom_refs = ref_energies[data.numbers]
            ref_sum = atom_refs.sum()
            if config.model.per_atom_target:
                data.energy = data.energy - ref_sum/data.num_atoms
            else:
                data.energy = data.energy - ref_sum

        # stress
        if stress_key is not None:
            if data.stress.dim() == 1:
                data.stress = voigt_6_to_full_3x3_stress(data.stress)
        yield data



def create_inmemory_dataset(data_generator, data_split = 'train', folder=None):
    data_list = []
    for data in tqdm(data_generator):
        data_list.append(data)
    if folder:
        os.makedirs(folder, exist_ok=True)
        torch.save(data_list, f'{folder}/data_{data_split}.pt')
        if data_split == 'train':
            energies = torch.tensor([data.energy for data in data_list])
            mean, std = float(energies.mean()), float(energies.std())
            normalizer_info = {'mean': mean, 'std': std}
            with open(f'{folder}/normalizer_params.json', "w") as f:
                json.dump(normalizer_info, f)
    return InMemoryDataset.from_data_list(data_list)
    

# not tested well
def create_lmdb_dataset(data_generator, lmdb_path, data_split = 'train',
                        map_size=1099511627776, chunk_size=5000):
    
    os.makedirs(lmdb_path + f'/data_{data_split}', exist_ok=True)

    env = lmdb.open(
        lmdb_path + f'/data_{data_split}',
        map_size=map_size,
        meminit=False,
        writemap=True,
        sync=False,
        metasync=False
    )
    
    buffer = BytesIO()
    total_count = 0
    txn = None  
    energies = []
    try:
        txn = env.begin(write=True)
        for idx, data in tqdm(enumerate(data_generator)):
            energies.append(data.energy)
            buffer.seek(0)
            torch.save(data, buffer, pickle_protocol=5, _use_new_zipfile_serialization=True)
            txn.put(f'graph_{idx}'.encode(), buffer.getvalue())
            buffer.truncate(0)
            total_count += 1
            if (idx + 1) % chunk_size == 0:
                txn.commit()
                txn = env.begin(write=True)

        if txn is not None:
            txn.put(b'num_graphs', pickle.dumps(total_count))
            txn.commit()
            
    except Exception as e:
        if txn is not None:
            txn.abort()
        raise e
        
    finally:
        buffer.close()
        env.close()

    env = lmdb.open(lmdb_path + f'/data_{data_split}')
    env.set_mapsize(env.info()['map_size'])  # Resize to actual usage
    env.close()
    if data_split == 'train':
        energies = np.array(energies)
        mean, std = energies.mean(), energies.std()
        normalizer_info = {'mean': float(mean), 'std': float(std)}
        with open(f'{lmdb_path}/normalizer_params.json', "w") as f:
            json.dump(normalizer_info, f)
    return LMDBDataset(lmdb_path + f'/data_{data_split}')



def collate_fn(batch: list[Data]) -> Data:

    collated = Data()

    def safe_cat(attr_name: str, dim=0, stack=False):
        tensors = [getattr(d, attr_name) for d in batch if getattr(d, attr_name) is not None]
        if not tensors:
            return
        if stack:
            setattr(collated, attr_name, torch.stack(tensors))
        else:
            setattr(collated, attr_name, torch.cat(tensors, dim=dim))

    safe_cat("node_attr", dim=0)
    safe_cat("positions", dim=0)
    safe_cat("numbers", dim=0)
    safe_cat("offset", dim=0)
    safe_cat("edge_attr", dim=0)
    safe_cat("edge_vector", dim=0)
    safe_cat("edge_length", dim=0)
    safe_cat("forces", dim=0)
    safe_cat("num_atoms", dim=0)
    
    safe_cat("cell", stack=True)
    safe_cat("energy", stack=True)
    safe_cat("stress", stack=True)
    

    node_counts = [0] + [d.positions.shape[0] for d in batch[:-1]]
    offsets = torch.tensor(node_counts).cumsum(0)
    
    edge_indices = []
    for d, offset in zip(batch, offsets):
        if d.edge_index is not None:
            edge_indices.append(d.edge_index + offset)
    
    if edge_indices:
        collated.edge_index = torch.cat(edge_indices, dim=1)

    collated.batch = torch.cat([
        torch.full((d.positions.shape[0],), i, dtype=torch.long) 
        for i, d in enumerate(batch)
    ], dim=0)

    return collated


def full_3x3_to_voigt_6_stress(stress_matrix):
    
    # Note: For stress, Voigt notation uses: 
    # σ11, σ22, σ33, σ23, σ13, σ12
    
    s11 = stress_matrix[..., 0, 0]
    s22 = stress_matrix[..., 1, 1]
    s33 = stress_matrix[..., 2, 2]
    
    s23 = stress_matrix[..., 1, 2]
    s13 = stress_matrix[..., 0, 2]
    s12 = stress_matrix[..., 0, 1]
    
    return torch.stack([s11, s22, s33, s23, s13, s12], dim=-1)

def voigt_6_to_full_3x3_stress(stress_vector):

    # handle both 2D (batch, 6) and 1D (6,) inputs
    original_shape = stress_vector.shape
    if len(original_shape) == 1:
        stress_vector = stress_vector.unsqueeze(0)
    
    # unbind the 6 components along the last dimension
    s1, s2, s3, s4, s5, s6 = torch.unbind(stress_vector, dim=-1)
    
    # Create the 3x3 stress tensor for each batch
    # Note: Voigt notation for stress: 
    # s4 = σ23, s5 = σ13, s6 = σ12
    stress_tensor = torch.stack([
        torch.stack([s1, s6, s5], dim=-1),
        torch.stack([s6, s2, s4], dim=-1),
        torch.stack([s5, s4, s3], dim=-1)
    ], dim=-2)
    
    # Remove batch dimension if input was 1D
    if len(original_shape) == 1:
        stress_tensor = stress_tensor.squeeze(0)
    
    return stress_tensor



class Normalizer(object):
    """From CGCNN by T. Xie"""

    def __init__(self, tensor = None):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = None if tensor is None else torch.mean(tensor)
        self.std = None if tensor is None else torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}
    
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
