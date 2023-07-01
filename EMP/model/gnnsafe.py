import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv
import numpy as np

class GNNSafe(nn.Module):
    '''
    The model class of energy-based models for out-of-distribution detection
    GNNSafe: args.use_reg = False, args.use_prop = True
    '''
    def __init__(self, in_feature, hidden_feature, num_classes, dropout):
        super(GNNSafe, self).__init__()
        self.dropout = dropout
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.num_classes = num_classes
        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.conv2 = GCNConv(hidden_feature, num_classes)

        self.bns = nn.BatchNorm1d(hidden_feature)

    def forward(self, x, edge_index):
        '''return predicted logits'''
        x = self.conv1(x, edge_index)
        x = self.bns(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.conv2(x, edge_index)
        return out

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        '''energy belief propagation, return the energy after propagation'''
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col, _ = edge_index.coo()
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)