import os
import json
import pickle
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lmdb
from ase.io import iread
from .data import Data, LMDBDataset, InMemoryDataset, atoms2data, DEFAULT_MAPPER



def build_dataloader(config, data_split):

    dataset = build_dataset(config, data_split)    
    loader = DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
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
            dataset = LMDBDataset(processed_data_path + f'/data_{data_split}')
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
    data_file = config['data'][f'data_{data_split}']
    energy_key = None if config['data']['energy_key'] is False else config['data']['energy_key']
    forces_key = None if config['data']['forces_key'] is False else config['data']['forces_key']
    stress_key = None if config['data']['stress_key'] is False else config['data']['stress_key']

    for atoms in iread(data_file):
        data = atoms2data(atoms,
               r_cut = config['data']['r_cut'],
               energy_key = energy_key,
               forces_key = forces_key,
               stress_key = stress_key,
               n_max_neighbors=None if config['data']['n_max_neighbors'] is False else config['data']['n_max_neighbors'],
               )
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
    


def create_lmdb_dataset(data_generator, lmdb_path, data_split = 'train', map_size=1099511627776, chunk_size=5000):
    
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



def get_atomic_types_mapper(config):

    if isinstance(config['atomic_types_mapper'], dict):
        return config['atomic_types_mapper']
    mapper_path = config['data']['processed_data_path'] + '/atomic_types_mapper.json'
    if os.path.exists(mapper_path):
        with open(mapper_path) as f:
            atomic_types_mapper = json.load(f)
    else:
        if config['atomic_types_mapper'] == 'default':
            atomic_types_mapper = DEFAULT_MAPPER
        elif config['atomic_types_mapper'] == 'from_scratch':
            unique_species = set()
            for atoms in tqdm(iread(config['data']['data_train']), desc = 'Getting atomic types mapper'):
                symbols = set(np.unique(atoms.symbols).tolist())
                unique_species.update(symbols - unique_species)
            unique_species = list(unique_species)
            unique_species.sort()
            atomic_types_mapper = dict(zip(unique_species, np.arange(len(unique_species)).tolist()))
    os.makedirs(config['data']['processed_data_path'], exist_ok=True)
    with open(mapper_path, "w")as f: 
        json.dump(atomic_types_mapper, f)
    return atomic_types_mapper



def collate_fn(batch):
    elem = batch[0]
    collated = Data()

    if elem.node_attr is not None:
        collated.node_attr = torch.cat([d.node_attr for d in batch], dim=0)
    collated.positions = torch.cat([d.positions for d in batch], dim=0)
    collated.numbers = torch.cat([d.numbers for d in batch], dim=0)
    
    if elem.offset is not None:
        collated.offset = torch.cat([d.offset for d in batch], dim=0)
    
    node_counts = torch.cumsum(torch.tensor([0] + [d.positions.shape[0] for d in batch[:-1]]), 0)
    collated.edge_index = torch.cat(
        [d.edge_index.to(torch.long) + offset for d, offset in zip(batch, node_counts)], dim=1)
    
    if elem.edge_attr is not None:
        collated.edge_attr = torch.cat([d.edge_attr for d in batch], dim=0)
    collated.edge_vector = torch.cat([d.edge_vector for d in batch], dim=0)
    collated.edge_length = torch.cat([d.edge_length for d in batch], dim=0)
    
    if elem.energy is not None:
        collated.energy = torch.stack([d.energy for d in batch])
    if elem.stress is not None:
        collated.stress = torch.stack([d.stress for d in batch])
    if elem.forces is not None:
        collated.forces = torch.cat([d.forces for d in batch], dim=0)
    
    collated.batch = torch.cat([torch.full((d.positions.shape[0],) , i) for i, d in enumerate(batch)], dim=0)
    return collated



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