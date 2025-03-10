import os
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import copy
import time
import torch
import sklearn.metrics
from model_GREMI import *

# Env
from utils import *

# DATA
loaded_data = torch.load('data.pt')
#
data_te = loaded_data['data_te']
te_omic = loaded_data['te_omic']
te_labels = loaded_data['te_labels']
exp_adj1 = loaded_data['exp_adj1']
exp_adj2 = loaded_data['exp_adj2']
exp_adj3 = loaded_data['exp_adj3']

te_dataset = torch.utils.data.TensorDataset(te_omic, te_labels)
te_data_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=4, shuffle=False)

input_in_dim = [1000,1000,503]
input_hidden_dim = [64]
network = Fusion(num_class=5, num_views=3, hidden_dim=input_hidden_dim, dropout=0.3, in_dim=input_in_dim)
checkpoint = torch.load('model-brca.pth')
network.load_state_dict(checkpoint['net'])
acc = checkpoint['acc']
epoch = checkpoint['epoch']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network.to(device)
loss_function = nn.CrossEntropyLoss()
network.eval()

test_loss = 0.0
test_corrects = 0
test_num = 0
output_y = torch.zeros(1,5)
pred_y = torch.zeros(1)
label_y = torch.zeros(1)
#
with torch.no_grad():
    for i, data in enumerate(te_data_loader, 0):
        batch_x, targets = data
        batch_x1 = batch_x[:, 0:1000].reshape(-1, 1000, 1)
        batch_x2 = batch_x[:, 1000:2000].reshape(-1, 1000, 1)
        batch_x3 = batch_x[:, 2000:].reshape(-1, 503, 1)
        batch_x1 = batch_x1.to(torch.float32)
        batch_x2 = batch_x2.to(torch.float32)
        batch_x3 = batch_x3.to(torch.float32)
        targets = targets.long()
        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)
        batch_x3 = batch_x3.to(device)
        targets = targets.to(device)
        exp_adj1 = exp_adj1.to(device)
        exp_adj2 = exp_adj2.to(device)
        exp_adj3 = exp_adj3.to(device)

        te_logits = network.infer(batch_x1, batch_x2, batch_x3, exp_adj1, exp_adj2, exp_adj3)
        te_prob = F.softmax(te_logits, dim=1)
        te_pre_lab = torch.argmax(te_prob, 1)

        output_y = torch.cat((output_y, te_logits.data.cpu()), dim=0)
        pred_y = torch.cat((pred_y, te_pre_lab.data.cpu()), dim=0)
        label_y = torch.cat((label_y, targets.data.cpu()), dim=0)
        test_corrects += torch.sum(te_pre_lab == targets.data)
        test_num += batch_x1.size(0)
    real_output_y = output_y[1:, :]
    real_pred_y = pred_y[1:]
    real_label_y = label_y[1:]
    test_acc = test_corrects.double().item() / test_num

    sk_acc = sklearn.metrics.accuracy_score(real_label_y, real_pred_y)
    sk_f1marco = sklearn.metrics.f1_score(real_label_y, real_pred_y, average='macro')
    sk_f1weighted = sklearn.metrics.f1_score(real_label_y, real_pred_y, average='weighted')
    print('acc : {:.8f}'.format(sk_acc))
    print('f1_weighted : {:.8f}'.format(sk_f1weighted))
    print('f1_marco : {:.8f}'.format(sk_f1marco))
    print('end')

