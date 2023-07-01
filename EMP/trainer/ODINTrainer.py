import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils.process import *
from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score, f1_score, average_precision_score
import numpy as np


class ODINTrainer:
    def __init__(self, args, dataset, model, optimizer, device):
        self.args = args
        self.dataset = dataset
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
    
    def train(self):
        best_t = self.args.max_epoch - 1
        best_metric = 0
        patience = 0
        for i in range(self.args.max_epoch):
            train_loss = self.train_single_epoch(i)

            if i % self.args.val_per_epoch == 0:
                current_metric, threshold = self.validation(i)
                if current_metric > best_metric:
                    best_t = i
                    patience = 0
                    best_metric = current_metric
                    #test
                    id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs = self.test(threshold)
                    # accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs = self.test()
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

        # calculate loss and update parameters
        self.optimizer.zero_grad()
        logits = self.model(x, adj)
        log_softmax = logits.log_softmax(dim=-1)
        loss = F.nll_loss(log_softmax[train_indices], y[train_indices])
        loss.backward()
        self.optimizer.step()

        # accuracy
        max_logsoft, pred = log_softmax[train_indices].max(1)
        acc = pred.eq(y[train_indices]).sum().item() / train_indices.size(0)

        print('Train | Epoch {} | train_loss: {}, train_acc: {}'.format(epoch, loss, acc))
        return loss.item()
        

    def validation(self, epoch, ood_label=-2):
        self.model.eval()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        x = Variable(x, requires_grad=True)

        y = self.dataset.get_labels().to(self.device)
        valid_indices =  self.dataset.get_val_indices()

        logits = self.model(x, adj, self.args.T)
        log_softmax = F.log_softmax(logits, dim=1)[valid_indices]
        temp_label = torch.argmax(log_softmax, -1)
        loss = F.nll_loss(log_softmax, temp_label)
        loss.backward()
        gradient = (torch.ge(x.grad[valid_indices].data, 0))
        gradient = (gradient.float() - 0.5) * 2
        tempInputs = x.detach().clone()

        tempInputs[valid_indices] = torch.add(tempInputs[valid_indices].data, gradient, alpha=-0.00001)
        logits = self.model(tempInputs, adj, self.args.T)
        log_softmax = F.log_softmax(logits, dim=1)

        max_logsoft, pred = log_softmax[valid_indices].max(1)
        max_softmax = torch.exp(max_logsoft)

        sorted_prob, sorted_index = torch.sort(max_softmax, descending=True) 

        # # cal threshold
        threshold = sorted_prob[round(valid_indices.size(0)*0.95)]

        ###
        # threshold = 0.5

        # cal classification accuracy
        # ood_mask = max_softmax < threshold
        # pred = pred * (~ood_mask) +(ood_label) * ood_mask
        accuracy = pred.eq(y[valid_indices]).sum().item() / valid_indices.size(0)
        # macro_f_score = f1_score(y[valid_indices[sorted_index[:round(valid_indices.size(0)*0.95)]]].cpu().detach().numpy(), pred[sorted_index[:round(valid_indices.size(0)*0.95)]].cpu().detach().numpy(), average="macro")
        macro_f_score = f1_score(y[valid_indices].cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro")

        print('Valid | Epoch {} | accuracy: {}, f1_score:{}'.format(epoch, accuracy, macro_f_score))
        return accuracy + macro_f_score, threshold


    def test(self, threshold, ood_label=-1):
        # threshold = 0.5
        self.model.eval()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        x = Variable(x, requires_grad=True)
        y = self.dataset.get_labels().to(self.device)
        test_id, test_ood = self.dataset.get_test_indices()
        test_indices = torch.cat([test_id, test_ood], 0)

        logits = self.model(x, adj, self.args.T)[test_indices]
        log_softmax = F.log_softmax(logits, dim=1)
        temp_label = torch.argmax(log_softmax, -1)

        loss = F.nll_loss(log_softmax, temp_label)
        loss.backward()
        gradient = (torch.ge(x.grad[test_indices].data, 0))
        gradient = (gradient.float() - 0.5) * 2
        tempInputs = x.detach().clone()
        tempInputs[test_indices] = torch.add(tempInputs[test_indices].data, gradient, alpha=-0.00001)
        logits = self.model(tempInputs, adj, self.args.T)
        log_softmax = F.log_softmax(logits, dim=1)
        max_logsoft, pred = log_softmax[test_indices].max(1)
        max_softmax = torch.exp(max_logsoft)

 
        # cal auroc and aupr
        ood_labels = torch.ones_like(y[test_indices])
        ood_labels[:test_id.size(0)] -= 1
        auroc = roc_auc_score(ood_labels.cpu().detach().numpy(), -max_softmax.cpu().detach().numpy())
        aupr_0 = average_precision_score(ood_labels.cpu().detach().numpy(), max_softmax.cpu().detach().numpy(), pos_label=0)
        aupr_1 = average_precision_score(ood_labels.cpu().detach().numpy(), -max_softmax.cpu().detach().numpy(), pos_label=1)
        fpr, tpr, thresholds = roc_curve(ood_labels.cpu().detach().numpy(), -max_softmax.cpu().detach().numpy(), drop_intermediate=False)
        f = fpr[abs((tpr - 0.95))<0.005].mean()
        if not np.isnan(f):
            fprs = f
        else:
            fprs = 0


         # cal classification accuracy
        id_accuracy = pred[:test_id.size(0)].eq(y[test_id]).sum().item() / test_id.size(0)
        id_f1 = f1_score(y[test_id].cpu().detach().numpy(), pred[:test_id.size(0)].cpu().detach().numpy(), average="macro")
        ood_mask = max_softmax < threshold
        pred = pred * (~ood_mask) +(ood_label) * ood_mask
        accuracy = pred.eq(y[test_indices]).sum().item() / test_indices.size(0)
        # accuracy = pred[:test_id.size(0)].eq(y[test_id]).sum().item() / test_id.size(0)
        macro_f_score = f1_score(y[test_indices].cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro")
        print('Test | accuracy: {}, f1_score:{}, auroc: {}, aupr_0: {}, aupr_1: {}, fprs: {}'.format(accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs))
        return id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs