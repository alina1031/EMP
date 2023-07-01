import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class GCN(nn.Module):
    '''
        input x shape is [node_num, node_feature]
        output out shape is [node_num, class_num]
    '''
    def __init__(self, in_feature, hidden_feature, num_classes, dropout):
        super(GCN, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.p = dropout
        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.conv2 = GCNConv(hidden_feature, num_classes)

    def forward(self, x, edge_index):
        edge_index = gcn_norm(edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.conv2(x, edge_index)
        return x

