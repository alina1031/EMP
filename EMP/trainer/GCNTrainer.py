import torch
import torch.nn.functional as F
from utils.process import *
from sklearn.metrics import roc_auc_score, auc, roc_curve, f1_score, average_precision_score
import numpy as np

class GCNTrainer:
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
                current_metric = self.validation(i)
                if current_metric > best_metric:
                    best_t = i
                    patience = 0
                    best_metric = current_metric
                    #test
                    accuracy, macro_f_score, max_softmax, pred, labels, embeddings = self.test()
                else:
                    patience += 1
                    if patience > 20:
                        break 
        np.save('result/Cora/GCN/plt/'+ self.args.dataset + '_labels.npy', labels)
        np.save('result/Cora/GCN/plt/'+ self.args.dataset + '_pred.npy', pred)
        np.save('result/Cora/GCN/plt/'+ self.args.dataset + '_prob.npy', max_softmax)
        np.save('result/Cora/GCN/plt/'+ self.args.dataset + '_embeddings.npy', embeddings)
        return best_t, accuracy, macro_f_score
    
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
        pred = logits[train_indices].max(1)[1]
        acc = pred.eq(y[train_indices]).sum().item() / train_indices.size(0)

        print('Train | Epoch {} | train_loss: {}, train_acc: {}'.format(epoch, loss, acc))
        return loss.item()

    @torch.no_grad()
    def validation(self, epoch):
        self.model.eval()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        y = self.dataset.get_labels().to(self.device)
        valid_indices =  self.dataset.get_val_indices()
        
        logits = self.model(x, adj)
        log_softmax = F.log_softmax(logits, dim=1)
        max_logsoft, pred = log_softmax[valid_indices].max(1)
        max_softmax = torch.exp(max_logsoft)
        sorted_prob, sorted_index = torch.sort(max_softmax, descending=True) 



        accuracy = pred.eq(y[valid_indices]).sum().item() / valid_indices.size(0)
        macro_f_score = f1_score(y[valid_indices].cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro")
        print('Valid | Epoch {} | accuracy: {}, f1_score:{}'.format(epoch, accuracy, macro_f_score))
        return accuracy + macro_f_score

    def test(self, ood_label=-1):
        self.model.eval()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        y = self.dataset.get_labels().to(self.device)
        test_id, test_ood = self.dataset.get_test_indices()
        test_indices = torch.cat([test_id, test_ood], 0)

        logits = self.model(x, adj)
        log_softmax = F.log_softmax(logits, dim=1)
        max_logsoft, pred = log_softmax[test_indices].max(1)
        max_softmax = torch.exp(max_logsoft)
 
       
        # cal accuracy
        accuracy = pred.eq(y[test_indices]).sum().item() / test_indices.size(0)
        macro_f_score = f1_score(y[test_indices].cpu().detach().numpy(), pred.cpu().detach().numpy(), average="macro")

        print('Test | accuracy: {}, f1_score:{}'.format(accuracy, macro_f_score))
        return accuracy, macro_f_score, max_softmax.cpu().detach().numpy(), pred.cpu().detach().numpy(), y.cpu().detach().numpy(), logits.cpu().detach().numpy()