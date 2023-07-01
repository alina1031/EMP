import torch
from torch import nn
import torch.nn.functional as F
import pdb
import numpy as np

def cal_discrimination_loss(entropy, train_indices):
    all_indices = torch.arange(0, entropy.size(0)).to(entropy.device)
    unlabeled = 1 - torch.zeros_like(all_indices).scatter(0, train_indices, 1)
    unlabeled_indices = torch.nonzero(unlabeled).squeeze()
    unlabeled_entropy = torch.index_select(entropy, 0, unlabeled_indices)
    discrimination_loss = torch.mean(unlabeled_entropy)
    return discrimination_loss    