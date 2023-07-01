import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch.autograd import Variable
import numpy as np

class ODIN(nn.Module):
    '''
        input x shape is [node_num, node_feature]
        output out shape is [node_num, class_num]
    '''
    def __init__(self, in_feature, hidden_feature, num_classes, dropout):
        super(ODIN, self).__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.p = dropout
        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.conv2 = GCNConv(hidden_feature, num_classes)
        self.bns = nn.BatchNorm1d(hidden_feature)

    def forward(self, x, edge_index, temperature=1.0):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bns(x)
        x = F.dropout(x, self.p, training=self.training)
        x = self.conv2(x, edge_index)
        return x/temperature