import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from data.cora import CoraDataSet
from data.computers import ComputersDataSet
from data.coauthor import CoauthorDataSet

from model.gcn import GCN
from model.gnnsafe import GNNSafe
from model.odin import ODIN
from model.emp import EMP
from trainer.GNNSAFETrainer import GNNSAFETrainer
from trainer.MSPTrainer import MSPTrainer
from trainer.ODINTrainer import ODINTrainer
from trainer.DOCTrainer import DOCTrainer
from trainer.GCNTrainer import GCNTrainer
from trainer.EMPTrainer import EMPTrainer
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    # data config
    parser.add_argument('--dataset', default = 'Cora')
    parser.add_argument('--ood_num', type=int, default=2)


    # model config
    parser.add_argument('--arch', type=str, default='gcn')
    parser.add_argument('--hidden', type=int, default= 256)
    #opgl: hidden:32

    # training config
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--val_per_epoch', type=int, default=5)
    # parser.add_argument('--seed', type=int, default=100)

    parser.add_argument('--layer_num', type=int, default=1)
    parser.add_argument('--threshold', type=str, default='k-means')
    parser.add_argument('--sample_times', type=int, default=50)
    parser.add_argument('--drop', type=int, default=1)
    #threshold:k-means, opgl, constant

    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--lu', type=float, default=2)
    parser.add_argument('--ll', type=float, default=4)

    parser.add_argument('--T', default=1000, type=float, help='temperature for Softmax')
    args = parser.parse_args()
    return args


def get_dataset(args, device):
    if args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'PubMed':
        #7 classes; 6 classes; 3 classes
        return CoraDataSet(dataset_name=args.dataset, ood_num=args.ood_num, device=device)
    elif args.dataset == 'Computers' or args.dataset == 'Photo':
        #10 classes ; 8 classes
        return ComputersDataSet(dataset_name=args.dataset, ood_num=args.ood_num, device=device)
    elif args.dataset == 'CS' or args.dataset == 'Physics':
        # 15 classes; 5 classes
        return CoauthorDataSet(dataset_name=args.dataset, ood_num=args.ood_num, device=device)
    else:
        raise NotImplementedError
    
def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)
    

def get_model(args, dataset):
    if args.arch == 'gcn':
        return GCN(in_feature=dataset.num_features, hidden_feature=args.hidden, num_classes=dataset.num_classes, dropout=args.dropout)
    elif args.arch == 'msp':
        return GCN(in_feature=dataset.num_features, hidden_feature=args.hidden, num_classes=dataset.num_classes, dropout=args.dropout)
    elif args.arch == 'odin':
        return ODIN(in_feature=dataset.num_features, hidden_feature=args.hidden, num_classes=dataset.num_classes, dropout=args.dropout)
    elif args.arch == 'doc':
        return GCN(in_feature=dataset.num_features, hidden_feature=args.hidden, num_classes=dataset.num_classes, dropout=args.dropout)
    elif args.arch == 'gnnsafe':
        return GNNSafe(in_feature=dataset.num_features, hidden_feature=args.hidden, num_classes=dataset.num_classes, dropout=args.dropout)
    elif args.arch == 'emp':
        return EMP(in_feature=dataset.num_features, hidden_feature=args.hidden, num_classes=dataset.num_classes, edge_num=dataset.edge_num, dropout=args.dropout)
    else:
        raise NotImplementedError
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = parse_args()
    seed_list = [100, 715, 442, 523, 665, 8, 287, 349, 307, 645]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.run == 1:
        set_seed(seed_list[0])
        dataset = get_dataset(args, device)
        model = get_model(args, dataset)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ## 1. 指数衰减：
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)

        if args.arch == 'gcn':
            Trainer = GCNTrainer(args, dataset, model, optimizer, device)    
        elif args.arch == 'msp':
            Trainer = MSPTrainer(args, dataset, model, optimizer, device)
        elif args.arch == 'odin':
            Trainer = ODINTrainer(args, dataset, model, optimizer, device)
        elif args.arch == 'doc':
            Trainer = DOCTrainer(args, dataset, model, optimizer, device)
        elif args.arch == 'gnnsafe':
            Trainer = GNNSAFETrainer(args, dataset, model, optimizer, device)
        elif args.arch == 'emp':
            Trainer = EMPTrainer(args, dataset, model, optimizer, device)
        else:
            raise NotImplementedError

        if args.arch == 'gcn':
            best_epoch, accuracy, macro_f_score= Trainer.train()
            print('Finally | best_epoch: {}, accuracy: {}, macro_f_score: {}'.format(best_epoch, accuracy*100, macro_f_score*100))
        else:
            best_epoch, id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs= Trainer.train()
            print('Finally | best_epoch: {}, accuracy: {}, macro_f_score: {}, auroc: {}, aupr_0:{}, aupr_1:{}, fprs: {}'.format(best_epoch, accuracy*100, macro_f_score*100, auroc*100, aupr_0*100, aupr_1*100, fprs*100))
    else:
        test_id_acc = []
        test_id_f1 = []
        test_acc = []
        test_f1 = []
        auroc = []
        aupr_0 = []
        aupr_1 = []
        fprs = []
        for iter in range(args.run):
            print(seed_list[iter])
            set_seed(seed_list[iter])
            dataset = get_dataset(args, device) 
            model = get_model(args, dataset)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            if args.arch == 'gcn':
                Trainer = GCNTrainer(args, dataset, model, optimizer, device)    
            elif args.arch == 'msp':
                Trainer = MSPTrainer(args, dataset, model, optimizer, device)
            elif args.arch == 'odin':
                Trainer = ODINTrainer(args, dataset, model, optimizer, device)
            elif args.arch == 'doc':
                Trainer = DOCTrainer(args, dataset, model, optimizer, device)
            elif args.arch == 'gnnsafe':
                Trainer = GNNSAFETrainer(args, dataset, model, optimizer, device)
            elif args.arch == 'emp':
                Trainer = EMPTrainer(args, dataset, model, optimizer, device)
            else:
                raise NotImplementedError
            
            if args.arch == 'gcn':
                best_epoch, accuracy_i, macro_f_score_i= Trainer.train()
                test_acc.append(accuracy_i*100)
                test_f1.append(macro_f_score_i*100)
            else:
                best_epoch, id_accuracy_i, id_f1_i, accuracy_i, macro_f_score_i, auroc_i, aupr_0_i, aupr_1_i, fprs_i= Trainer.train()
                test_id_acc.append(id_accuracy_i*100)
                test_id_f1.append(id_f1_i*100)
                test_acc.append(accuracy_i*100)
                test_f1.append(macro_f_score_i*100)
                auroc.append(auroc_i*100)
                aupr_0.append(aupr_0_i*100)
                aupr_1.append(aupr_1_i*100)
                fprs.append(fprs_i*100)
        
        if args.arch == 'gcn':
            acc_mean = np.mean(test_acc)
            acc_std = np.std(test_acc, ddof=1)
            f1_mean = np.mean(test_f1)
            f1_std = np.std(test_f1,ddof=1)

            print(acc_mean)
            print(acc_std)
            print(f1_mean)
            print(f1_std)
        else: 
            id_acc_mean = np.mean(test_id_acc)
            id_acc_std = np.std(test_id_acc, ddof=1)
            id_f1_mean = np.mean(test_id_f1)
            id_f1_std = np.std(test_id_f1,ddof=1)
            acc_mean = np.mean(test_acc)
            acc_std = np.std(test_acc, ddof=1)
            f1_mean = np.mean(test_f1)
            f1_std = np.std(test_f1,ddof=1)
            auroc_mean = np.mean(auroc)
            auroc_std = np.std(auroc,ddof=1)
            aupr_0_mean = np.mean(aupr_0)
            aupr_0_std = np.std(aupr_0,ddof=1)
            aupr_1_mean = np.mean(aupr_1)
            aupr_1_std = np.std(aupr_1,ddof=1)
            fprs_mean = np.mean(fprs)
            fprs_std = np.std(fprs,ddof=1)
            
            print(test_id_acc)
            print(test_id_f1)
            print(test_acc)
            print(test_f1)
            print(auroc)
            print(aupr_0)
            print(aupr_1)
            print(fprs)
            

            print(id_acc_mean)
            print(id_acc_std)
            print(id_f1_mean)
            print(id_f1_std)
            print(acc_mean)
            print(acc_std)
            print(f1_mean)
            print(f1_std)
            print(auroc_mean)
            print(auroc_std)
            print(aupr_0_mean)
            print(aupr_0_std)
            print(aupr_1_mean)
            print(aupr_1_std)
            print(fprs_mean)
            print(fprs_std)


