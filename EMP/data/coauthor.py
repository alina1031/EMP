import os.path as osp
import numpy as np
import torch
from torch_geometric.datasets import Coauthor
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split

import random
from collections import Counter

class CoauthorDataSet():
    def __init__(self, dataset_name, ood_num, ood_label=-1, split_ratio=[0.6, 0.2, 0.2], device='cuda'):
        self.name = dataset_name
        self.ood_num = ood_num
        self.ood_label = ood_label
        self.split_ratio = split_ratio
        self.use_device = device

        path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', self.name)
        transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
        dataset = Coauthor(path, self.name, transform=transform)
        self.data = dataset[0]
        self.adj = self.data.adj_t
        self.row, self.col, _ = self.data.adj_t.coo()
        self.edge_num = self.row.size(0)

        self.split_ood(dataset)
        self.special_train_test_split(self.y_true)

        # print(dataset.num_features)
        self.num_features = dataset.num_features
        self.num_classes = torch.max(self.y_true).item() + 1

        self.log()

    def split_ood(self, dataset):
        original_num_classes = dataset.num_classes
        label_count = Counter(dataset.data.y.numpy())
        sort_count = sorted(label_count.items(),key = lambda item:item[1],reverse=True)
        id_categories = []
        for item in sort_count:
            id_categories.append(item[0])
        ood_categories = id_categories[-self.ood_num:]
        for item in ood_categories:
            id_categories.remove(item)

        self.y_true = self.reassign_labels(self.data.y, id_categories, ood_categories)
        self.id_categories = id_categories
        self.ood_categories = ood_categories
        
    def reassign_labels(self, y, id_categories, ood_categories):
        y = y.numpy()
        old_new_label_dict = {old_label:new_label for new_label, old_label in enumerate(id_categories)}
        def convert_label(old_label):
            if old_label in old_new_label_dict:
                return old_new_label_dict[old_label]
            elif old_label in ood_categories:
                return self.ood_label

        new_y = [
            convert_label(label) for label in y
        ]
        new_y = np.array(new_y)

        return torch.from_numpy(new_y)

    def special_train_test_split(self, y_true):
        y_true = y_true.numpy()
        id_indices = np.where(y_true >= 0)[0]
        ood_indices = np.where(y_true == self.ood_label)[0]

        id_train_indices, id_val_test_indices = train_test_split(id_indices, test_size=1-self.split_ratio[0])
        id_val_indices, id_test_indices = train_test_split(id_val_test_indices, test_size=self.split_ratio[2] / (1-self.split_ratio[0]))

        self.train_indices = torch.tensor(id_train_indices, device=self.use_device)
        self.valid_indices = torch.tensor(id_val_indices, device=self.use_device)
        self.test_indices = np.concatenate([id_test_indices, ood_indices], axis=0)
        self.test_indices = torch.tensor(self.test_indices, device=self.use_device)

        self.test_id = torch.tensor(id_test_indices, device=self.use_device)
        self.test_ood = torch.tensor(ood_indices, device=self.use_device)

    def get_inputs(self):
        return self.data.x, self.adj
    
    def get_features(self):
        return self.data.x
    
    def get_nodesnum(self):
        return self.adj.size(0)

    def get_eddge(self):
        return self.adj.to_dense().sum().int().item()
    
    def get_edgeindex(self):
        edge_index = torch.nonzero(self.adj.to_dense())
        return edge_index

    def get_labels(self):
        return self.y_true

    def get_train_indices(self):
        return self.train_indices
    
    def get_val_indices(self):
        return self.valid_indices
    
    def get_test_indices(self):
        return self.test_id, self.test_ood
    
    def get_size(self, indice):
        return indice.size(0)
    
    def get_id_categories(self):
        return self.id_categories
    
    def log(self):
        train_size = self.get_size(self.train_indices)
        valid_size = self.get_size(self.valid_indices)
        test_id_size = self.get_size(self.test_id)
        test_ood_size = self.get_size(self.test_ood)
        total_size = self.get_size(self.y_true)

        print('Dataset | Name: {}, class_num {} example {{id_categories: {}, ood_categories : {}}}'.format(self.name, self.num_classes, self.id_categories, self.ood_categories))
        print('Dataset | total {} example {{train: {}, valid: {}, test_id: {}, test_ood: {}}}'.format(self.name, train_size, valid_size,test_id_size, test_ood_size))