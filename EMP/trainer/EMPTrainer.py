import pdb
import torch
import torch.nn.functional as F
from utils.process import *
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, average_precision_score
import numpy as np
from sklearn.cluster import KMeans

class EMPTrainer:
    def __init__(self, args, dataset, model, optimizer, device):
        self.args = args
        self.dataset = dataset
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.y_true = self.dataset.get_labels()

    def train(self):
        best_t = self.args.max_epoch - 1
        best_metric = 0
        patience = 0

        for i in range(self.args.max_epoch):
            train_loss = self.train_single_epoch(i)

            if i>0 and i % self.args.val_per_epoch == 0:
                current_metric, threshold = self.validation(i)
                if current_metric > best_metric:
                    best_t = i
                    patience = 0
                    best_metric = current_metric
                    # test
                    id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs, att_drop, pred, ood_score, labels = self.test(i, threshold)
                else:
                    patience += 1
                    if patience > 50:
                        break 
        return best_t, id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs
        

    def train_single_epoch(self, epoch):
        self.model.train()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        y = self.dataset.get_labels().to(self.device)
        train_indices = self.dataset.get_train_indices()
        label = y[train_indices]

        # calculate loss and update parameters
        self.optimizer.zero_grad()
        logits, edge_index = self.model(x, adj, self.args.drop)

        # classification_loss
        log_softmax = logits.log_softmax(dim=-1)
        classification_loss = F.nll_loss(log_softmax[train_indices], label)

        # discrimination_loss
        entropy = (torch.exp(log_softmax) * log_softmax).sum(1)
        if self.args.layer_num == 0:
            discrimination_loss_L = torch.mean(-entropy[train_indices])
            discrimination_loss_U = cal_discrimination_loss(entropy, train_indices)
        else:
            entropy_after_pro = self.model.propagation(entropy, edge_index, self.args.layer_num, 0.5)
            discrimination_loss_L = torch.mean(-entropy_after_pro[train_indices])
            discrimination_loss_U = cal_discrimination_loss(entropy_after_pro, train_indices)

        loss = classification_loss + self.args.lu * discrimination_loss_U + self.args.ll * discrimination_loss_L

        loss.backward()
        self.optimizer.step()
        
        print('Train | Epoch {} | train_loss: {}, classification_loss: {}, discrimination_loss_U: {}, discrimination_loss_L: {}'.format(epoch, loss, classification_loss, discrimination_loss_U, discrimination_loss_L))
        return loss.item()

    @torch.no_grad()
    def validation(self, epoch, ood_label = -2):
        self.model.eval()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        y = self.dataset.get_labels().to(self.device)
        valid_indices = self.dataset.get_val_indices()

        probs, _ = self.model.val_forward(x, adj, self.args.sample_times, self.args.drop)
        max_softmax, pred = probs[valid_indices].max(1)
        
        entropy = (probs * torch.log(probs)).sum(1)
        if self.args.layer_num == 0:
            ood_score = entropy
        else:
            ood_score = self.model.propagation(entropy, adj, self.args.layer_num, 0.5, 'val')[valid_indices]
        # threshold
        if self.args.threshold == 'opgl':
            avg_seen = ood_score.mean()
            probs_sort, _ = torch.sort(ood_score, descending=False)
            E_unseen_probs = probs_sort[:probs_sort.size(0)//10]
            avg_E_unseen = E_unseen_probs.mean()
            threshold = (avg_seen + avg_E_unseen) / 2.0
        elif self.args.threshold == 'contant':
            threshold = (torch.max(ood_score) + torch.min(ood_score)) / 2.0
        else:
            threshold = None

        # accuracy
        accuracy = pred.eq(y[valid_indices]).sum().item() / valid_indices.size(0)
        macro_f_score = f1_score(y[valid_indices].cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro")

        print('Valid | Epoch {} | accuracy: {}, f1_score:{}'.format(epoch, accuracy, macro_f_score))
        return accuracy + macro_f_score, threshold

    @torch.no_grad()
    def test(self, epoch, threshold, ood_label=-1):
        self.model.eval()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        y = self.dataset.get_labels().to(self.device)
        test_id, test_ood = self.dataset.get_test_indices()
        test_indices =  torch.cat([test_id, test_ood], 0)


        probs, drop_edge = self.model.val_forward(x, adj, self.args.sample_times, self.args.drop)

        max_softmax, pred = probs[test_indices].max(1)

        test_entropy = (probs * torch.log(probs)).sum(1)
        # test_ood_score = (probs * torch.log(probs)).sum(1)[test_indices]
        if self.args.layer_num == 0:
            ood_score = test_entropy
        else:
            ood_score = self.model.propagation(test_entropy, adj, self.args.layer_num, 0.5, 'test')
       
        test_ood_score = ood_score[test_indices]
        test_ood_score = torch.nan_to_num(test_ood_score, nan=0.0, posinf=0.0, neginf=0.0)


        # cal auroc and aupr
        ood_labels = torch.ones_like(y[test_indices])
        ood_labels[:test_id.size(0)] -= 1
        auroc = roc_auc_score(ood_labels.cpu().detach().numpy(), -test_ood_score.cpu().detach().numpy())

        aupr_0 = average_precision_score(ood_labels.cpu().detach().numpy(), test_ood_score.cpu().detach().numpy(), pos_label=0)
        aupr_1 = average_precision_score(ood_labels.cpu().detach().numpy(), 1-test_ood_score.cpu().detach().numpy(), pos_label=1)

        fpr, tpr, thresholds = roc_curve(ood_labels.cpu().detach().numpy(), 1-test_ood_score.cpu().detach().numpy(), drop_intermediate=False)
        f = fpr[abs((tpr - 0.95))<0.005].mean()
        if not np.isnan(f):
            fprs = f
        else:
            fprs = 0

        
        # cal classification accuracy
        id_accuracy = pred[:test_id.size(0)].eq(y[test_id]).sum().item() / test_id.size(0)
        id_f1 = f1_score(y[test_id].cpu().detach().numpy(), pred[:test_id.size(0)].cpu().detach().numpy(), average="macro")
        
        if self.args.threshold == 'k-means':
            ood_kmeans = KMeans(n_clusters=2).fit(-test_ood_score.cpu().detach().numpy().reshape(-1, 1))
            ood_pred = ood_kmeans.predict(-test_ood_score.cpu().detach().numpy().reshape(-1, 1))
            if ood_kmeans.cluster_centers_[0] > ood_kmeans.cluster_centers_[1]:
                ood_pred = 1 - ood_pred
            test_pred = pred.cpu().detach().numpy()
            label = y[test_indices].cpu().detach().numpy()
            pred = test_pred * (1-ood_pred) +(ood_label) * ood_pred
            accuracy = (pred==label).sum() / pred.shape[0]
            macro_f_score = f1_score(label, pred, average="macro")
        elif self.args.threshold == 'opgl' or self.args.threshold == 'contant':        
            ood_mask = test_ood_score < threshold
            pred = pred * (~ood_mask) +(ood_label) * ood_mask
            accuracy = pred.eq(y[test_indices]).sum().item() / test_indices.size(0)
            # accuracy = pred[:test_id.size(0)].eq(y[test_id]).sum().item() / test_id.size(0)
            macro_f_score = f1_score(y[test_indices].cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro")


        print('Test | accuracy: {}, f1_score:{}, auroc: {}, aupr_0: {}, aupr_1: {}, fprs: {}'.format(accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs))
        return id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs, drop_edge.cpu().detach().numpy(), pred, test_ood_score.cpu().detach().numpy(), y.cpu().detach().numpy()