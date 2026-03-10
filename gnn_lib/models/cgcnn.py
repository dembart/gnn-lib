import torch
from torch import nn
import torch.nn.functional as F
from ..scatter_utils import scatter
from .utils import get_symmetric_displacement, compute_forces_virials

    

class AtomicEmbedding(nn.Module):

    def __init__(self, num_node_fea, node_dim, norm=False):
        super(AtomicEmbedding, self).__init__()
        # inut_dim is the maximum atomic number in the dataset
        self.embedding = nn.Embedding(num_node_fea, node_dim)
        self.layer_norm = nn.LayerNorm(node_dim) if norm else nn.Identity()

    def forward(self, x):
        embed = self.embedding(x)
        embed = self.layer_norm(embed)
        return embed


        
class GaussianFilter(nn.Module):

    def __init__(self, r_min=1.0, r_max=6.0, edge_dim=64):

        super(GaussianFilter, self).__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.edge_dimtures = edge_dim
        _filter = torch.linspace(r_min, r_max, edge_dim)
        step = (r_max - r_min) / edge_dim
        self.register_buffer("_filter", _filter)
        self.register_buffer("step", torch.tensor(step))

    def forward(self, edge_lengths):

        edge_lengths = edge_lengths.to(self._filter.device)
        diff = edge_lengths[:, None] - self._filter
        out = torch.exp(-0.5 * torch.square(diff / self.step))
        return out



class OLP(nn.Module):

    def __init__(self, in_dim, out_dim, norm=False, dropout=0.0):
        super(OLP, self).__init__()
        self.norm = norm
        self.activation = nn.Softplus()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if norm else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x



class GraphConvLayer(nn.Module):

    def __init__(self, node_dim, edge_dim, reduce='add', dropout=0.0):
        super(GraphConvLayer, self).__init__()
        assert reduce in ['add', 'mean']
        self.reduce = reduce
        self.lin_f = nn.Linear(2 * node_dim + edge_dim, node_dim)
        self.lin_s = nn.Linear(2 * node_dim + edge_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_attrs, edge_index, edge_attrs):

        source, target = edge_index
        x_i = node_attrs[source]  
        x_j = node_attrs[target] 
        z = torch.cat([x_i, x_j, edge_attrs], dim=-1)  
        z = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        message = scatter(z, source, dim=0, reduce=self.reduce, dim_size=node_attrs.size(0))
        out = node_attrs + self.norm(message)
        return out
    


class CrystalGraphConvNet(nn.Module):

    def __init__(self,
                 in_dim=128,
                 E0s=None,
                 edge_dim=64,
                 node_dim=64,
                 num_conv_layers=3,
                 num_hidden_layers=2,
                 hidden_dim=128,
                 reduce_messages='add',
                 per_atom_target=True,
                 norm_hidden_layers=True,
                 norm_node_embed=False,
                 r_min=1.0,
                 r_max=6.0,
                 dropout=0.1,
                 compute_forces=False,
                 compute_stress=False,
                 ):

        super(CrystalGraphConvNet, self).__init__()

        if E0s is not None:
            ref_energies = torch.zeros(in_dim)
            for z, e0 in E0s.items():
                ref_energies[z] = e0
            self.register_buffer('ref_energies', ref_energies)
        else:
            self.ref_energies = None
        self.compute_forces=compute_forces
        self.compute_stress=compute_stress
        self.per_atom_target = per_atom_target
        self.embedding = AtomicEmbedding(in_dim, node_dim, norm=norm_node_embed)
        self.gaussian_filter = GaussianFilter(r_min=r_min, r_max=r_max, edge_dim=edge_dim)

        self.conv_layers = [GraphConvLayer(node_dim,
                                           edge_dim,
                                           reduce=reduce_messages,
                                           ) for _ in range(num_conv_layers)]
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.linear_layers = [OLP(node_dim, hidden_dim, norm=norm_hidden_layers)]
        

        
        if num_hidden_layers > 1:
            # For all layers except the last one
            for _ in range(num_hidden_layers - 1):
                self.linear_layers.append(OLP(hidden_dim, hidden_dim,
                                              norm=norm_hidden_layers, dropout=dropout))
            # for _ in range(1, num_hidden_layers - 1):
            #     self.linear_layers.append(OLP(hidden_dim, hidden_dim,
            #                                   norm=norm_hidden_layers, dropout=dropout))
        self.linear_layers.append(nn.Linear(hidden_dim, 1))
        self.linear_layers = nn.ModuleList(self.linear_layers)


    
    def _energy_from_e0s_and_residuals(self, energy, data):
    
        if self.ref_energies is None:
            return energy
        
        atom_refs = self.ref_energies[data.numbers]  # (n_atoms,)
    
        # sum atomic reference energies per batch
        unique_batches = torch.unique(data.batch)
        ref_sum = torch.zeros(len(unique_batches), device=energy.device)
        ref_sum.index_add_(0, data.batch, atom_refs)
        
        # Reshape ref_sum to match energy dimensions
        if energy.dim() > 1:
            ref_sum = ref_sum.unsqueeze(-1)  # (batch_size, 1)
    
        if self.per_atom_target:
            # Convert energy from per-atom to total energy
            if energy.dim() > 1:
                # energy is per-atom, so multiply by num_atoms to get total energy
                raw = energy + ref_sum / data.num_atoms.unsqueeze(-1)
            else:
                raw = energy + ref_sum / data.num_atoms
        else:
            raw = energy + ref_sum
        return raw

        
    
    def forward(self, data):
        
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.numbers.shape[0],
                                     dtype=int).to(data.numbers.device)
    
        if data.cell.dim() == 2:  # [3,3] -> [1,3,3]
            data.cell = data.cell.unsqueeze(0)
    
        if self.compute_forces or self.compute_stress:
            with torch.enable_grad():
                
                positions = data.positions.clone().requires_grad_(True)
                
                n_graphs = data.batch.max().item() + 1
                
                positions, shifts, displacement = get_symmetric_displacement(
                    positions,
                    data.offset,
                    data.cell,
                    data.edge_index,
                    n_graphs,
                    data.batch
                )
                     
                source, target = data.edge_index
                vectors = positions[target] + shifts - positions[source]
                edge_length = torch.linalg.norm(vectors, dim=-1)

                node_embeddings = self.embedding(data.numbers)
                edge_attrs = self.gaussian_filter(edge_length)
                
                for conv in self.conv_layers:
                    node_embeddings = conv(node_embeddings, data.edge_index, edge_attrs)
                
                for olp in self.linear_layers:
                    node_embeddings = olp(node_embeddings)
                    
                energy = scatter(node_embeddings, data.batch, dim=0,
                                 reduce='mean' if self.per_atom_target else 'add')

                if self.per_atom_target:
                    total_energy = energy * data.num_atoms.view(-1, 1)
                else:
                    total_energy = energy

                forces, virials, stress = compute_forces_virials(
                                            total_energy,
                                            positions,
                                            displacement,
                                            data.cell,
                                            self.training,
                                            self.compute_stress,
                                            )
        else:
            forces = None
            stress = None
            positions = data.positions
            edge_length = data.edge_length
            node_embeddings = self.embedding(data.numbers)
            edge_attrs = self.gaussian_filter(edge_length)
            
            for conv in self.conv_layers:
                node_embeddings = conv(node_embeddings, data.edge_index, edge_attrs)
            
            for olp in self.linear_layers:
                node_embeddings = olp(node_embeddings)
            reduce = 'mean' if self.per_atom_target else 'add'
            energy = scatter(node_embeddings, data.batch, dim=0, reduce=reduce)
        
        return {'energy': energy, 'forces': forces, 'stress': stress}

    def size(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path): 
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=False))
    
    def from_checkpoint(self, path):
        self.load_state_dict(torch.load(path, weights_only=False)['model_state'])