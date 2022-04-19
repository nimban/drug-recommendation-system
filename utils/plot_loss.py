import matplotlib.pyplot as plt
import json
import numpy as np
import os

base_path = 'loss_curve.json'
exec_id = '2022-01-17 12:10:42'
batch_size = 4

with open("outputs/loss_curve.json") as file:
    losses = json.load(file)

# plt.plot(losses['train_loss'], label='train_loss')
# plt.plot(losses['val_loss'], label='val_loss')
# plt.plot(losses['val_f1'], label='val_f1')
plt.plot(losses['val_precision'], label='val_precision')
plt.plot(losses['val_recall'], label='val_recall')

plt.legend()
plt.show()