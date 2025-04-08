import torch
from torch import nn
import torch.nn.functional as F
from ase.data import atomic_numbers
from gnn_lib.scatter_utils import scatter
from .base_model import BaseModel



class OneHot(nn.Module):
    
    def __init__(self, atomic_types_mapper=None):
        super().__init__()
        self.mapper = {atomic_numbers[key]: value for key, value in atomic_types_mapper.items()}
        max_atomic_num = max(self.mapper.keys())
        mapping = torch.zeros(max_atomic_num + 1, dtype=torch.long)
        for num, idx in self.mapper.items():
            mapping[num] = idx
        self.register_buffer('_precomputed_mapping', mapping)
        
        self.num_classes = len(atomic_types_mapper)
        self.register_buffer("_eye_matrix", torch.eye(self.num_classes))

    def forward(self, numbers):
        class_indices = self._precomputed_mapping[numbers.to(self._eye_matrix.device)]
        return self._eye_matrix[class_indices]



class GaussianFilter(nn.Module):

    def __init__(self, r_min=1.0, r_max=6.0, num_edge_fea=64):

        super(GaussianFilter, self).__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.num_edge_features = num_edge_fea
        _filter = torch.linspace(r_min, r_max, num_edge_fea)
        step = (r_max - r_min) / num_edge_fea
        self.register_buffer("_filter", _filter)
        self.register_buffer("step", torch.tensor(step))

    def forward(self, edge_lengths):

        if not isinstance(edge_lengths, torch.Tensor):
            edge_lengths = torch.tensor(edge_lengths, dtype=torch.float32)
        edge_lengths = edge_lengths.to(self._filter.device)
        diff = edge_lengths[:, None] - self._filter
        out = torch.exp(-0.5 * torch.square(diff / self.step))
        return out
    


class OLP(nn.Module):

    def __init__(self, in_dim, out_dim,
                  batch_norm=False, activation=nn.Softplus()):
        super(OLP, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.linear = nn.Linear(in_dim, out_dim)
        if self.batch_norm:
            self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, x):

        x = self.linear(x)
        if self.batch_norm:
            x = self.norm(x)
        x = self.activation(x)
        return x



class NodeEmbedding(nn.Module):

    def __init__(self, num_node_fea, num_embed_fea, batch_norm=False, activation=None, bias=True):
        super(NodeEmbedding, self).__init__()

        self.batch_norm = batch_norm
        self.activation = activation
        self.linear = nn.Linear(num_node_fea, num_embed_fea, bias=bias)
        if self.batch_norm:
            self.norm = nn.BatchNorm1d(num_embed_fea)

    def forward(self, node_attrs):
        node_embeddings = self.linear(node_attrs)
        if self.batch_norm:
            node_embeddings = self.norm(node_embeddings)
        if self.activation is not None:
            node_embeddings = self.activation(node_embeddings)
        return node_embeddings



class GraphConvLayer(nn.Module):

    def __init__(self, node_fea_len, edge_fea_len, reduce='add'):
        super(GraphConvLayer, self).__init__()

        if reduce not in ['add', 'mean']:
            raise ValueError("reduce must be 'add' or 'mean'")
        self.reduce = reduce
        self.lin_f = nn.Linear(2 * node_fea_len + edge_fea_len, node_fea_len)
        self.lin_s = nn.Linear(2 * node_fea_len + edge_fea_len, node_fea_len)
        self.norm = nn.BatchNorm1d(node_fea_len)

    def forward(self, node_attrs, edge_index, edge_attrs):

        source, target = edge_index
        x_i = node_attrs[source]  
        x_j = node_attrs[target] 
        z = torch.cat([x_i, x_j, edge_attrs], dim=-1)  
        z = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        message = scatter(z, source, dim=0, reduce=self.reduce, dim_size=node_attrs.size(0))
        out = node_attrs + self.norm(message)
        return out
    


class CrystalGraphConvNet(BaseModel):

    def __init__(self,
                 atomic_types_mapper,
                 num_edge_fea=64,
                 num_embed_fea=64,
                 n_conv=3,
                 n_h = 2,
                 h_fea_len=128,
                 reduce_messages = 'add',
                 reduce_nodes = 'mean',
                 batch_norm_hidden_layers = True,
                 batch_norm_node_embed = False,
                 r_min = 1.0,
                 r_max = 6.0
                 ):

        super(CrystalGraphConvNet, self).__init__()
        
        self.onehot_encoder = OneHot(atomic_types_mapper)
        self.embedding = NodeEmbedding(len(atomic_types_mapper), num_embed_fea,
                                       batch_norm = batch_norm_node_embed)
        self.gaussian_filter = GaussianFilter(r_min=r_min,
                                              r_max=r_max,
                                              num_edge_fea=num_edge_fea)
        self.n_h = n_h
        self.reduce_messages = reduce_messages
        self.reduce_nodes = reduce_nodes
        
        self.conv_layers = [GraphConvLayer(num_embed_fea,
                                           num_edge_fea,
                                           reduce = self.reduce_messages,
                                           ) for _ in range(n_conv)]
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.linear_layers = [OLP(num_embed_fea, h_fea_len, batch_norm=batch_norm_hidden_layers)]
        if self.n_h > 1:
            for _ in range(1, self.n_h - 1):
                self.linear_layers.append(OLP(h_fea_len, h_fea_len, batch_norm = batch_norm_hidden_layers))
        self.linear_layers.append(nn.Linear(h_fea_len, 1))
        self.linear_layers = nn.ModuleList(self.linear_layers)



    def forward(self, data):
        
        # embedding -> norm -> conv -> linear -> scatter

        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.node_attr.shape[0], dtype = int).to(data.node_attr.get_device())
        node_attrs = self.onehot_encoder(data.numbers)
        node_embeddings = self.embedding(node_attrs)
        edge_attrs = self.gaussian_filter(data.edge_length)
        for conv in self.conv_layers:
            node_embeddings = conv(node_embeddings, data.edge_index, edge_attrs)
        for olp in self.linear_layers:
            node_embeddings = olp(node_embeddings)
        out = scatter(node_embeddings, data.batch, dim=0, reduce = self.reduce_nodes)
        return out
    

    
    @classmethod
    def from_config(cls, atomic_types_mapper, config, load_checkpoint = False):
        model = cls(atomic_types_mapper, **config['model_params'])
        if load_checkpoint:
            checkpoint = torch.load(config['logging']['checkpoint_dir'] + "/best_checkpoint.pt")
            model.load_state_dict(checkpoint["model_state"])
        return model
    

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()