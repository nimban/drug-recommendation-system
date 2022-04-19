'''
Script to build, compile, train Gate model
'''

import pickle5
import torch as torch
import torch.nn as nn
import json
import numpy as np
import os
import random

from collections import Counter
# from pymagnitude import *

from sklearn.model_selection import train_test_split

from model import GATE
from utils_lib import to_device, CREATE_EVENT_DICT

cuda = torch.device('cuda')

print(torch.__version__)

'''
Load model Inputs from File
'''
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

## Create Test Train Split

patient_list = patient_dict.keys()

X = [patient for patient in patient_list]

sample = random.sample(X, 1)

code_dict = {idx: code for code, idx in label_dict.items()}

####  Define Hyperparamters and initilize model layers

model = GATE(c)
model = to_device(model)

model.load_state_dict(torch.load('outputs/2022-01-17 12:10:42/model_weights'))

full_patient_history = patient_dict[sample[0]]

pred, sigged, Y_hat = model(full_patient_history)
pred_mask = pred>0
for idx, med in enumerate(pred_mask):
    if med:
        print(code_dict[idx])
