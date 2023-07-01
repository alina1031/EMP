import pdb
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
# from torch_geometric.nn import GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree

from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch_geometric.utils import dense_to_sparse

from torch_sparse import SparseTensor, matmul



class EMP(nn.Module):
    def __init__(self, in_feature, hidden_feature, num_classes, edge_num, dropout=0.0):
        super(EMP, self).__init__()
        self.p = dropout
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.num_classes = num_classes

        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.conv2 = GCNConv(hidden_feature, num_classes)
        self.bns = nn.BatchNorm1d(hidden_feature)

        self.drop_edge = nn.Parameter(torch.empty(edge_num), requires_grad=True)
        nn.init.normal_(self.drop_edge, mean=0, std=1)

    
    def forward(self, x, edge_index, drop, temperature=0.1):
        # pdb.set_trace()
        if drop == 0:
            row, col, _ = edge_index.coo()
            edge_index = SparseTensor(row=row, col=col, value= torch.ones(row.size(0)).to(edge_index.device()),sparse_sizes=(edge_index.size(0), edge_index.size(1)))
            h = F.relu(self.conv1(x, edge_index))
            h1 = F.dropout(h, self.p, training=self.training)
            out = self.conv2(h1, edge_index)
            return out, edge_index
        edge_index = self.dropout_edge(edge_index, self.drop_edge, temperature)
        h = F.relu(self.conv1(x, edge_index))

        h1 = F.dropout(h, self.p, training=self.training)
        out = self.conv2(h1, edge_index)
        return out, edge_index
    
    def val_forward(self, x, edge_index, sample_times, drop, temperature=0.1):
        ### cal dropedge prob:
        if drop == 0:
            edge_index = gcn_norm(edge_index)
            h = F.relu(self.conv1(x, edge_index))
            h1 = F.dropout(h, self.p, training=self.training)
            out = self.conv2(h1, edge_index)
            output = F.softmax(out, -1)
            return output, self.drop_edge.sigmoid()
        output = []
        if sample_times == 0:
            edge_index = self.dropout_edge(edge_index, self.drop_edge, temperature)
            edge_index = gcn_norm(edge_index)
            h = F.relu(self.conv1(x, edge_index))
            h1 = F.dropout(h, self.p, training=self.training)
            out = self.conv2(h1, edge_index)
            output = F.softmax(out, -1)
            return output, self.drop_edge.sigmoid()
        for i in range(sample_times):
            edge_index_i = self.dropout_edge(edge_index, self.drop_edge, temperature)
            edge_index_i = gcn_norm(edge_index_i)
            h = F.relu(self.conv1(x, edge_index_i))
            h1 = F.dropout(h, self.p, training=self.training)
            out = self.conv2(h1, edge_index_i)
            output.append(F.softmax(out, -1))
        output = torch.stack(output, -1)
        output = output.mean(-1)
        return output, self.drop_edge.sigmoid()
    
    def dropout_edge(self, edge_index, drop_edge, temperature=0.1):
        row, col, _ = edge_index.coo()
        drop_edge = drop_edge.sigmoid()
        weighted_adjacency_matrix = RelaxedBernoulli(temperature=torch.Tensor([temperature]).to(edge_index.device()),
                                                     probs=drop_edge).rsample()
        eps = 0.5
        mask = (weighted_adjacency_matrix > eps).detach().float()
        weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)

        # edge_mask = (weighted_adjacency_matrix > eps).to(edge_index.device())
        new_edge_index = SparseTensor(row=row, col=col, value=weighted_adjacency_matrix ,sparse_sizes=(edge_index.size(0), edge_index.size(1)))
        return new_edge_index
    
    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5, mode='train'):
        '''energy belief propagation, return the energy after propagation'''
        e = e.unsqueeze(1)
        N = e.shape[0]
        # print(edge_index)
        row, col, val = edge_index.coo()
        E = torch.zeros(N, N).to(self.drop_edge)
        # drop_edge = torch.ones_like(row).float()
        # print(col)
        if mode == 'train':
            E[row, col] = self.drop_edge.sigmoid() * val
            E[col, row] = self.drop_edge.sigmoid() * val
        else:
            E[row, col] = self.drop_edge.sigmoid()
            E[col, row] = self.drop_edge.sigmoid()
        value = E / (torch.sum(E, dim=1, keepdim=True) + 1e-9)
        # pdb.set_trace()
        value = value[col, row]
        
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)