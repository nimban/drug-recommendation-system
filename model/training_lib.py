'''
Training and Validation method definition for a single epoch
'''

import torch as torch
import torch.nn as nn
import random
from tqdm.auto import tqdm
import numpy as np
# from pymagnitude import *
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, jaccard_score
from optuna import TrialPruned

from utils_lib import to_device
cuda = torch.device('cuda')


def hinge_loss(y_hat, y_true):
    threshold = nn.Threshold(threshold=0, value=0)
    true_inds = [ind for ind, value in enumerate(y_true) if value==1]
    hl = 0
    for ind in true_inds:
        hl += torch.sum(threshold(1 - (y_hat[ind] - y_hat))) / len(true_inds)
    return hl

# def hinge_loss(y_hat, y_true):
#     threshold = nn.Threshold(threshold=0, value=0)
#     all_true_inds = [[ind for ind, value in enumerate(y_true_e) if value==1] for y_true_e in y_true]
#     hl = 0
#     for true_inds in all_true_inds:
#         for ind in true_inds:
#             hl += torch.sum(threshold(1 - (y_hat[true_inds][ind] - y_hat))) / len(true_inds)
#     return hl


def train(X_train, patient_dict, embeddings, gam, tdu, classifier, optimizer,
          criterion, gamma1, gamma2, gamma3, ddi_adj, c):
    embeddings.train()
    gam.train()
    tdu.train()
    classifier.train()
    train_set_size = len(X_train)
    train_iter = 0
    losses = 0
    np.random.seed(0)
    random.seed(0)
    X_shuffled = np.random.permutation(X_train)
    avg_loss = []
    ddi_loss = []
    sigmoid = nn.Sigmoid()
    precs, recalls, f1s, rocs, aprs, jacc, lrs = [], [], [], [], [], [], []
    for idx, patient_id in tqdm(enumerate(X_shuffled), total=len(X_shuffled)):
        # print('patient-', patient_id)
        if idx % 1000 == 0:
            torch.cuda.empty_cache()
        full_patient_history = patient_dict[patient_id]
        #full_patient_history = to_device(full_patient_history)
        visit_count = len(full_patient_history)
        if visit_count < 2:
            continue
        # print('admissions len-', visit_count)
        loss = to_device(torch.tensor(0, dtype=torch.float, requires_grad=True))  # to device ?
        for i in range(2, visit_count+1):
            train_iter += 1
            # print('\nsubset till-', i)
            hiddens = tdu.initHidden(c)
            Ft_1 = to_device(hiddens[0])
            F_hat_t_1 = to_device(hiddens[1])
            visits_subset = full_patient_history[0:i-1]
            for j, visit_j in enumerate(visits_subset):
                if j < i-1:
                    adj, code_indices = to_device(visit_j[0]), to_device(visit_j[1])
                else:
                    adj, code_indices = to_device(visit_j[2]), to_device(visit_j[3])
                #adj_gpu = to_device(adj)
                #code_indices_gpu = to_device(code_indices)
                gam_in = embeddings(code_indices)       # Convert Codes to  node embeddings
                gam_output = gam(gam_in, adj)           # Incorporate Adjacency matrix / neighbour data into node embedding (same size)
                F_hat_t_1, Ft_1  = tdu(gam_output, Ft_1, F_hat_t_1, code_indices)
                #F_hat_t_1, Ft_1 = F_hat_t, Ft
            #gam_output = to_device(gam_output)
            #F_hat_t = to_device(F_hat_t)
            Ot = torch.cat([gam_output, F_hat_t_1[code_indices]], dim=1)
            Y_hat = classifier(Ot)
            sigged = sigmoid(Y_hat)
            pred = (sigged > classifier.thresh).float()
            #print('pred on cuda - ', pred.is_cuda())
            #print('ddi on cuda - ', ddi_adj.is_cuda())
            #ddi_l = torch.sum(pred.view(-1, 1) * ddi_adj.float() @ pred.view(-1, 1)) // 2
            #ddi_loss.append(ddi_l)
            last = to_device(full_patient_history[-1][-1])
            loss = loss + gamma1 * criterion(Y_hat, last) + gamma3 * hinge_loss(sigged, last)
            l = float(loss.item())
        # del F_hat_t_1, Ft_1, gam_in, gam_output, Ot, Y_hat, sigged
        pred = pred.cpu().detach()
        #l = l / (visit_count-1)
        losses += l
        optimizer.zero_grad()
        avg_loss.append(l)
        loss.backward()
        optimizer.step()
        del loss
        lrs.append([param['lr'] for param in optimizer.param_groups][0])
    train_loss = losses / train_iter
    return train_loss, lrs


def val(X_val, patient_dict, embeddings, gam, tdu, classifier, criterion, epoch,
        gamma1, gamma2, gamma3, ddi_adj, c):
    embeddings.eval()
    gam.eval()
    tdu.eval()
    classifier.eval()
    sigmoid = nn.Sigmoid()
    val_set_size = len(X_val)
    sigmoid = nn.Sigmoid()
    precs, recalls, f1s, rocs, aprs, jacc = [], [], [], [], [], []
    l_total = 0
    #test_loss = to_device(torch.tensor(0, dtype=torch.float, requires_grad=False))
    print('Val size =', len(X_val))
    with torch.no_grad():
        for idx, patient_id in tqdm(enumerate(X_val)):
            if idx % 1000 == 0:
                torch.cuda.empty_cache()
            full_patient_history = patient_dict[patient_id]
            if len(full_patient_history) < 2:
                continue
            test_loss = to_device(torch.tensor(0, dtype=torch.float, requires_grad=True))
            hiddens = tdu.initHidden(c)
            Ft_1 = to_device(hiddens[0])
            F_hat_t_1 = to_device(hiddens[1])
            for i, visit_i in enumerate(full_patient_history):
                if i < len(full_patient_history) - 1:
                    adj, code_indices = to_device(visit_i[0]), to_device(visit_i[1])
                else:
                    adj, code_indices = to_device(visit_i[2]), to_device(visit_i[3])
                gam_in = embeddings(code_indices)  # Convert Codes to  node embeddings
                gam_output = gam(gam_in, adj)  # Incorporate Adjacency matrix / neighbour data into node embedding (same size)
                F_hat_t_1, Ft_1 = tdu(gam_output, Ft_1, F_hat_t_1, code_indices)
                #F_hat_t_1, Ft_1 = F_hat_t, Ft
            #test = to_device(F_hat_t_1)
            Ot = torch.cat([gam_output, F_hat_t_1[code_indices]], dim=1)
            Y_hat = classifier(Ot)
            sigged = sigmoid(Y_hat)
            pred = (sigged > classifier.thresh).float()
            last = to_device(full_patient_history[-1][-1])
            test_loss = gamma1 * criterion(Y_hat, last) + gamma3 * hinge_loss(sigged,last)
                        #gamma2 * (torch.sum(pred.view(-1, 1) * ddi_adj.float() @ pred.view(-1, 1)) // 2)
            l = float(test_loss.item())
            l_total += l
            pred = pred.cpu().detach()
            Y_true = full_patient_history[-1][-1]
            try:
                precs.append(precision_score(Y_true, pred))
                recalls.append(recall_score(Y_true, pred))
                f1s.append(f1_score(Y_true, pred))
                rocs.append(roc_auc_score(Y_true, sigged.cpu().detach()))
                aprs.append(average_precision_score(Y_true, sigged.cpu().detach()))
                jacc.append(jaccard_score(Y_true, pred))
            except:
                print("it effed")
                # raise TrialPruned()
            del test_loss, adj, code_indices, gam_in, gam_output, F_hat_t_1, Ft_1, Ot, Y_hat, sigged, last
    val_loss = l_total / val_set_size
    print("Epoch {} Validation loss: {}".format(epoch, val_loss))
    print("Precision: {}".format(np.mean(precs)))
    print("Recall: {}".format(np.mean(recalls)))
    print("F1: {}".format(np.mean(f1s)))
    print("AUROC: {}".format(np.mean(rocs)))
    print("APR: {}".format(np.mean(aprs)))
    print("JACC: {}".format(np.mean(jacc)))
    return val_loss, np.mean(precs), np.mean(recalls), np.mean(rocs), np.mean(aprs), np.mean(f1s)
