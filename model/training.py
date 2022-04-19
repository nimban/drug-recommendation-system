'''
Script to build, compile, train Gate model
'''

import pickle5
import torch as torch
import torch.nn as nn
import json
import numpy as np
import os
import datetime

from collections import Counter
# from pymagnitude import *

from sklearn.model_selection import train_test_split

from model import GATE
from gate_train_lib import train, val
from utils_lib import to_device

cuda = torch.device('cuda')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print(torch.__version__)

'''
Load model Inputs from File
'''

directory = str(datetime.datetime.now())[0:-7]
base_path = os.path.join('/home/nn122/remote_runtime/recommender/outputs', directory)
os.mkdir(base_path)

features_base_path = 'data/features'

patient_codes_cleaned_path = os.path.join(features_base_path, 'patient_codes_cleaned.pickle')        # All Codes grouped by Patient/Admission
M_matrix_path = os.path.join(features_base_path, 'all_codes_pmi_matrix.pickle')
event_location_dict_path = os.path.join(features_base_path, 'event_location_dict.pickle')
med_labels_path = os.path.join(features_base_path, 'medications_labels.pickle')
patient_dict_path = os.path.join(features_base_path, 'patient_dict.pickle')

with open(patient_dict_path, 'rb') as f:
    patient_dict = pickle5.load(f)

with open(med_labels_path, 'rb') as f:
    label_dict = pickle5.load(f)       # label_dict TO label_dict

# with open(base_path + 'recs_add.pickle', 'rb') as file:
#     recs_add = pickle5.load(file)

with open(patient_codes_cleaned_path, 'rb') as file:
    combined = pickle5.load(file)

with open(event_location_dict_path, 'rb') as file:
    event_location_dict = pickle5.load(file)

with open('data/input/ddi_adj.pickle', 'rb') as file:
    ddi_adj = pickle5.load(file)

c = len(event_location_dict)

"""
Define Train and Test Loop
"""

## Create Test Train Split

patient_list = patient_dict.keys()

X = [patient for patient in patient_list]
y = [len(patient_dict[patient]) for patient in patient_list]

only_once = [item for item in y if y.count(item) == 1]

occurs = Counter()
for ndc in combined['NDC']:
    for d in ndc:
        occurs[d] += 1

X.extend([patient for patient in patient_list if len(patient_dict[patient]) in only_once])

y = [len(patient_dict[patient]) for patient in patient_list]

y.extend([len(patient_dict[patient]) for patient in patient_list if len(patient_dict[patient]) in only_once])

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y,
                                                    test_size=0.2)


####  Define Hyperparamters and initilize model layers

gam_dropout = 0.38450344118714536
tdu_dropout = 0.061361479721181944
ddi_dropout = 0.06234617470113224
gamma1 = 0.7262516915120416
gamma2 = 0.6869345549100704
gamma3 = 0.40230067999770214
num_heads = 2
lr = 0.00021152207011365955
weight_decay = 0.01083088261894025
beta1 = 0.9402433442497843
beta2 = 0.9748917373718193

criterion = to_device(nn.BCEWithLogitsLoss(reduction='mean'))

gate = GATE(c)
gate = to_device(gate)

parameters = [{"params":gate.embeddings.parameters()}, {"params":gate.gam.parameters()}, \
          {"params":gate.tdu.parameters()}, {"params": gate.miml.parameters()}]

optimizer = torch.optim.Adam(parameters, lr=lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=1, T_mult=2, eta_min=1e-6)

ddi_adj = np.pad(ddi_adj, 99)
ddi_adj = torch.Tensor(ddi_adj)
to_device(ddi_adj)

##   Training Loop


EPOCHS = 50
train_losses = []
val_losses = []
val_precs = []
val_recs = []
val_aurocs = []
val_aprs = []
val_f1s = []
lrs = []

# X_train = X_train[0:100]
# X_test = X_test[0:10]

losses = {
    'train_loss': [],
    'val_loss': [],
    'val_f1': [],
    'val_precision': [],
    'val_recall': [],
}


def save_state_to_file(losses, model):
    with open(os.path.join(base_path, 'loss_curve.json'), 'w') as convert_file:
        convert_file.write(json.dumps(losses))
    torch.save(model.state_dict(), os.path.join(base_path, 'model_weights'))


for epoch in range(0, EPOCHS):
    X_t = np.squeeze(X_train)
    print('\nTraining Epoch ', epoch, ' ...')
    train_loss = train(X_t, patient_dict, gate, optimizer, criterion, gamma1, gamma2, gamma3, ddi_adj)
    losses['train_loss'].append(train_loss[0])
    lrs.append(train_loss[1])
    X_val = np.squeeze(X_test)
    print('\nTesting...')
    val_loss = val(X_val, patient_dict, gate, criterion, epoch, gamma1, gamma2, gamma3, ddi_adj)
    losses['val_loss'].append(val_loss[0])
    losses['val_f1'].append(val_loss[-1])
    losses['val_precision'].append(val_loss[1])
    losses['val_recall'].append(val_loss[2])
    save_state_to_file(losses, gate)
