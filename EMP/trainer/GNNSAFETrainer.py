import torch
import torch.nn.functional as F
from utils.process import *
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix
from torch_sparse import fill_diag
import numpy as np
import math
from sklearn.cluster import KMeans

class GNNSAFETrainer:
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
                current_metric = self.validation(i)
                if current_metric > best_metric:
                    best_t = i
                    patience = 0
                    best_metric = current_metric
                    # test
                    id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs, energy, pred, label = self.test(i)
                else:
                    patience += 1
                    if patience > 50:
                        break
                np.save('result/Cora/GNNSafe/'+ self.args.dataset + '_labels.npy', label)
        np.save('result/Cora/GNNSafe/'+ self.args.dataset + '_pred.npy', pred)
        np.save('result/Cora/GNNSafe/'+ self.args.dataset + '_score.npy', energy)
        
        # results = []
        # for i in range(self.args.max_epoch):
        #     train_loss = self.train_single_epoch(i)

        #     if i>0 and i % self.args.val_per_epoch == 0:
        #         valid_loss = self.validation(i)
        #         # test
        #         auroc, aupr_0, aupr_1, fprs, aupr = self.test(i)
        #         results.append([i, valid_loss, auroc, aupr_0, aupr_1, fprs, aupr])
        # results = torch.tensor(results)
        # valid_loss_list = results[:, 1]
        # np.save('valid_loss.npy', valid_loss_list)
        # best_t = valid_loss_list.argmin().item()
        # auroc = results[best_t, 2]
        # aupr_0 = results[best_t, 3]
        # aupr_1 = results[best_t, 4]
        # fprs = results[best_t, 5]
        # aupr = results[best_t, 6]
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
        logits = self.model(x, adj)

        # classification_loss
        log_softmax = logits.log_softmax(dim=-1)
        classification_loss = F.nll_loss(log_softmax[train_indices], label)

        
        loss = classification_loss
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

        max_softmax, pred = torch.exp(log_softmax[train_indices]).max(1)

        # accuracy
        accuracy = pred.eq(y[train_indices]).sum().item() / train_indices.size(0)
        macro_f_score = f1_score(y[train_indices].cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro")
        
        print('Train | Epoch {} | train_loss: {}, train_acc: {}, train_f1: {}'.format(epoch, loss, accuracy, macro_f_score))
        return loss.item()

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        y = self.dataset.get_labels().to(self.device)
        valid_indices = self.dataset.get_val_indices()

        logits = self.model(x, adj)

        # calculate valid loss
        log_softmax = logits.log_softmax(dim=-1)
        valid_loss = F.nll_loss(log_softmax[valid_indices], y[valid_indices])

        max_softmax, pred = torch.exp(log_softmax[valid_indices]).max(1)

        # accuracy
        accuracy = pred.eq(y[valid_indices]).sum().item() / valid_indices.size(0)
        macro_f_score = f1_score(y[valid_indices].cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro")

        print('Valid | Epoch {} | accuracy: {}, f1_score:{}, valid_loss: {}'.format(epoch, accuracy, macro_f_score, valid_loss))
        # return valid_loss.cpu().detach().numpy()
        return accuracy + macro_f_score

    @torch.no_grad()
    def test(self, epoch, ood_label=-1):
        self.model.eval()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        y = self.dataset.get_labels().to(self.device)
        test_id, test_ood = self.dataset.get_test_indices()
        test_indices =  torch.cat([test_id, test_ood], 0)

        logits = self.model(x ,adj)

        # neg_energy = torch.logsumexp(logits, dim=-1)
        
        probs = F.softmax(logits, dim=1)
        max_softmax, pred = probs[test_indices].max(1)
        test_entropy = (probs * torch.log(probs)).sum(1)
        test_ood_score = self.model.propagation(test_entropy, adj, 2, 0.5)[test_indices]
        test_ood_score = torch.nan_to_num(test_ood_score, nan=0.0, posinf=0.0, neginf=0.0)

        # test_ood_score = self.model.propagation(neg_energy, adj, 2, 0.5)[test_indices]


        # cal auroc and aupr
        ood_labels = torch.ones_like(y[test_indices])
        ood_labels[:test_id.size(0)] -= 1


        auroc = roc_auc_score(ood_labels.cpu().detach().numpy(), -test_ood_score.cpu().detach().numpy())

        # aupr = average_precision_score(ood_labels.cpu().detach().numpy(), test_ood_score.cpu().detach().numpy())

        aupr_0 = average_precision_score(ood_labels.cpu().detach().numpy(), test_ood_score.cpu().detach().numpy(), pos_label=0)
        aupr_1 = average_precision_score(ood_labels.cpu().detach().numpy(), -test_ood_score.cpu().detach().numpy(), pos_label=1)

        fpr, tpr, thresholds = roc_curve(ood_labels.cpu().detach().numpy(), -test_ood_score.cpu().detach().numpy(), drop_intermediate=False)
        f = fpr[abs((tpr - 0.95))<0.005].mean()
        if not np.isnan(f):
            fprs = f
        else:
            fprs = 0

        ood_kmeans = KMeans(n_clusters=2).fit(-test_ood_score.cpu().detach().numpy().reshape(-1, 1))
        ood_pred = ood_kmeans.predict(-test_ood_score.cpu().detach().numpy().reshape(-1, 1))
        if ood_kmeans.cluster_centers_[0] > ood_kmeans.cluster_centers_[1]:
            ood_pred = 1 - ood_pred

        # cal classification accuracy
        id_accuracy = pred[:test_id.size(0)].eq(y[test_id]).sum().item() / test_id.size(0)
        id_f1 = f1_score(y[test_id].cpu().detach().numpy(), pred[:test_id.size(0)].cpu().detach().numpy(), average="macro")
        # ood_mask = test_ood_score < threshold
        test_pred = pred.cpu().detach().numpy()
        label = y[test_indices].cpu().detach().numpy()
        pred = test_pred * (1-ood_pred) +(ood_label) * ood_pred
        accuracy = (pred==label).sum() / pred.shape[0]
        # accuracy = pred[:test_id.size(0)].eq(y[test_id]).sum().item() / test_id.size(0)
        macro_f_score = f1_score(label, pred, average="macro")

        print('Test | accuracy: {}, f1_score:{}, auroc: {}, aupr_0: {}, aupr_1: {}, fprs: {}'.format(accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs))
        return id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs, test_ood_score.cpu().detach().numpy(), pred, label
    

        # print('Test | auroc: {}, aupr_0: {}, aupr_1: {}, fprs: {}'.format(auroc, aupr_0, aupr_1, fprs))
        # return auroc, aupr_0, aupr_1, fprs, test_ood_score.cpu().detach().numpy()