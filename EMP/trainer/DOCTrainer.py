import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score, f1_score, average_precision_score
import numpy as np
from scipy.stats import norm as dist_model


# #fit a gaussian model
def fit(prob_pos_X):
    prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
    pos_mu, pos_std = dist_model.fit(prob_pos)
    return pos_mu, pos_std

class DOCTrainer:
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
                current_metric, mu_stds = self.validation(i)
                if current_metric > best_metric:
                    best_t = i
                    patience = 0
                    best_metric = current_metric
                    #test
                    id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs = self.test(mu_stds)
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
        train_indices = self.dataset.get_train_indices().to(self.device)
        id_class = self.dataset.get_id_categories()

        # calculate loss and update parameters
        self.optimizer.zero_grad()
        logits = self.model(x, adj)
        train_prob = torch.sigmoid(logits)[train_indices]
        train_labels = torch.zeros_like(train_prob)
        for i in range(len(id_class)):
            train_labels[y[train_indices]==i,i] = 1

        loss = F.binary_cross_entropy(train_prob, train_labels)

        loss.backward()
        # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5, norm_type=2)
        self.optimizer.step()

        # accuracy
        pred = train_prob.max(1)[1]
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
        id_class = torch.unique(y[valid_indices])

        logits = self.model(x, adj)
        val_prob = torch.sigmoid(logits)[valid_indices]
        val_pred = val_prob.max(1)[1]

        ##fit a gaussian model
        #calculate mu, std of each id class
        mu_stds = []
        for i in range(len(id_class)):
            pos_mu, pos_std = fit(val_prob[y[valid_indices]==i, i].cpu().numpy())
            mu_stds.append([pos_mu, pos_std])
        
        # accuracy
        accuracy = val_pred.eq(y[valid_indices]).sum().item() / valid_indices.size(0)
        macro_f_score = f1_score(y[valid_indices].cpu().detach().numpy(), val_pred.cpu().detach().numpy(), average="macro")

        print('Valid | Epoch {} | accuracy: {}, f1_score:{}'.format(epoch, accuracy, macro_f_score))
        return accuracy + macro_f_score, mu_stds

    @torch.no_grad()
    def test(self, mu_stds, ood_label=-1):
        self.model.eval()
        x, adj = self.dataset.get_inputs()
        x, adj = x.to(self.device), adj.to(self.device)
        y = self.dataset.get_labels().to(self.device)
        test_id, test_ood = self.dataset.get_test_indices()
        test_indices = torch.cat([test_id, test_ood], 0).to(self.device)
        id_class = self.dataset.get_id_categories()

        logits = self.model(x, adj)
        test_prob = torch.sigmoid(logits)[test_indices]
        test_y_pred = []
        scale = 1.
        for p in test_prob.cpu().numpy():# loop every test prediction
            max_class = np.argmax(p)# predicted class
            max_value = np.max(p)# predicted probability
            threshold = max(0.5, mu_stds[max_class][0] - scale * mu_stds[max_class][1])#find threshold for the predicted class
            # if threshold != 0.5:
                # print(threshold)
            # print("threshold:", threshold)
            if max_value >= threshold:
                test_y_pred.append(max_class)#predicted probability is greater than threshold, accept
            else:
                test_y_pred.append(ood_label)#otherwise, reject
        
        # test_y_gt = torch.where(y[test_indices]<0, len(id_class), y[test_indices])
        test_y_pred = torch.tensor(test_y_pred).to(self.device)

 
        # cal auroc and aupr
        ood_labels = torch.ones_like(y[test_indices])
        ood_labels[:test_id.size(0)] -= 1
        auroc = roc_auc_score(ood_labels.cpu().detach().numpy(), - test_prob.max(1)[0].cpu().detach().numpy())
        aupr_0 = average_precision_score(ood_labels.cpu().detach().numpy(),  test_prob.max(1)[0].cpu().detach().numpy(), pos_label=0)
        aupr_1 = average_precision_score(ood_labels.cpu().detach().numpy(), - test_prob.max(1)[0].cpu().detach().numpy(), pos_label=1)

        fpr, tpr, thresholds = roc_curve(ood_labels.cpu().detach().numpy(), -test_prob.max(1)[0].cpu().detach().numpy(), drop_intermediate=False)
        f = fpr[abs((tpr - 0.95))<0.005].mean()
        if not np.isnan(f):
            fprs = f
        else:
            fprs = 0


        # cal classification accuracy
        id_accuracy = test_y_pred[:test_id.size(0)].eq(y[test_id]).sum().item() / test_id.size(0)
        id_f1 = f1_score(y[test_id].cpu().detach().numpy(), test_y_pred[:test_id.size(0)].cpu().detach().numpy(), average="macro")

        accuracy = test_y_pred.eq(y[test_indices]).sum().item() / test_indices.size(0)
        macro_f_score = f1_score(y[test_indices].cpu().detach().numpy(), test_y_pred.cpu().detach().numpy(), average="macro")
        # print(test_y_pred)
        print('Test | accuracy: {}, f1_score:{}, auroc: {}, aupr_0: {}, aupr_1: {}, fprs: {}'.format(accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs))
        return id_accuracy, id_f1, accuracy, macro_f_score, auroc, aupr_0, aupr_1, fprs