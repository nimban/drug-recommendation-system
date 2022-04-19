import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import os

# base_path = '/home/nn122/remote_runtime/newfold/ipa_refine/outputs/'
# exec_id = '2022-01-03 17:26:25'
# batch_size = 4

def plot_losses(losses):
    plt.plot(losses['val_fape'], label='val_fape')
    plt.plot(losses['train_fape'], label='train_fape')
    plt.legend()
    plt.show()


if torch.cuda.is_available():
    print("cuda available")
    global cuda
    cuda = torch.device('cuda')
else:
    print("cuda not found")


def to_device(tensor):
    if torch.cuda.is_available():
        if isinstance(tensor, list):
            tensor = [t.to(cuda) for t in tensor]
        else:
            tensor = tensor.to(cuda)
    return tensor


class CREATE_EVENT_DICT(object):
    """
    Records the indices of any medical event. For example if the following dataset has the input:
    -------------------------
        PATIENT  |  DRUG
    -------------------------
            0    |  [A1, B2]
    -------------------------
            1    |  [A2, B1]
    -------------------------
    Then it will return the following event dictionary:
    {'A1':0, 'B2': 1, 'A2': 2, 'B1': 3}
    You want the inverse of this
    {0: 'A1', ....}
    """

    def __init__(self, event_type, dataset):
        self.event_type = event_type
        self.dataset = dataset
        self.event_col = dataset.columns.get_loc(self.event_type)

    def event_to_idx(self):
        self.event_to_idx_ = {}
        event_to_idx_ = self.event_to_idx_
        dataset = self.dataset
        for idx, row in dataset.iterrows():
            event_codes = row[self.event_col]
            for event in event_codes:
                if event not in event_to_idx_:
                    event_to_idx_[event] = len(event_to_idx_)

        return self.event_to_idx_

    def idx_to_event(self):
        event_to_idx_ = self.event_to_idx()
        reverse_idx = {idx: code for code, idx in event_to_idx_.items()}

        return reverse_idx
