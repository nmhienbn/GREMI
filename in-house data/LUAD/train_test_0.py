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
from model_GREMI import *

# Env
from utils import *

# DATA
loaded_data = torch.load('data.pt')
#
data_tr = loaded_data['data_tr']
tr_omic = loaded_data['tr_omic']
tr_labels = loaded_data['tr_labels']
data_te = loaded_data['data_te']
te_omic = loaded_data['te_omic']
te_labels = loaded_data['te_labels']
exp_adj1 = loaded_data['exp_adj1']
exp_adj2 = loaded_data['exp_adj2']
exp_adj3 = loaded_data['exp_adj3']
#DATA LOADRE
tr_dataset = torch.utils.data.TensorDataset(tr_omic, tr_labels)
tr_data_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=32, shuffle=True)
te_dataset = torch.utils.data.TensorDataset(te_omic, te_labels)
te_data_loader = torch.utils.data.DataLoader(dataset=te_dataset, batch_size=32, shuffle=False)


num_epochs = 3000
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loss_function = nn.CrossEntropyLoss()
input_in_dim = [200,200,200]
input_hidden_dim = [64]
network = Fusion(num_class=2, num_views=3, hidden_dim=input_hidden_dim, dropout=0.1, in_dim=input_in_dim)
network.to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)

best_model_wts = copy.deepcopy(network.state_dict())
best_acc = 0.0
best_epoch = 0
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []

# Early Stopping
patience = 50
early_stopping_counter = 0

for epoch in range(0, num_epochs):
    isPrint = (epoch % 100 == 0)
    # Print epoch
    if isPrint:
        print(' Epoch {}/{}'.format(epoch, num_epochs - 1))
        print("-" * 10)
    # Set current loss value
    network.train()
    current_loss = 0.0
    train_loss = 0.0
    train_corrects = 0
    train_num = 0

    for i, data in enumerate(tr_data_loader, 0):

        batch_x, targets = data
        batch_x1 = batch_x[:, 0:200].reshape(-1, 200, 1)
        batch_x2 = batch_x[:, 200:400].reshape(-1, 200, 1)
        batch_x3 = batch_x[:, 400:].reshape(-1, 200, 1)

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

        optimizer.zero_grad()
        loss_fusion, tr_logits, gat_output1, gat_output2, gat_output3, output1, output2, output3 = network(batch_x1, batch_x2, batch_x3, exp_adj1, exp_adj2, exp_adj3, targets)
        tr_prob = F.softmax(tr_logits, dim=1)
        tr_pre_lab = torch.argmax(tr_prob, 1)

        loss = loss_fusion
        loss.backward()
        optimizer.step()


        train_loss += loss.item() * batch_x1.size(0)
        train_corrects += torch.sum(tr_pre_lab == targets.data)
        train_num += batch_x1.size(0)
    # Evaluationfor this fold
    network.eval()
    test_loss = 0.0
    test_corrects = 0
    test_num = 0
    for i, data in enumerate(te_data_loader, 0):
        batch_x, targets = data
        batch_x1 = batch_x[:, 0:200].reshape(-1, 200, 1)
        batch_x2 = batch_x[:, 200:400].reshape(-1, 200, 1)
        batch_x3 = batch_x[:, 400:].reshape(-1, 200, 1)
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

        test_corrects += torch.sum(te_pre_lab == targets.data)
        test_num += batch_x1.size(0)

    train_loss_all.append(train_loss / train_num)
    train_acc_all.append(train_corrects.double().item() / train_num)
    test_acc_all.append(test_corrects.double().item() / test_num)
    if isPrint:
        print('{} Train Loss : {:.8f} Train ACC : {:.8f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{}  Test ACC : {:.8f}'.format(epoch, test_acc_all[-1]))

    if test_acc_all[-1] > best_acc:
        best_acc = test_acc_all[-1]
        best_epoch = epoch + 1
        best_model_wts = copy.deepcopy(network.state_dict())
        early_stopping_counter = 0
        # Saving the model
        save_path = f'/kaggle/working/model-luad1.pth'
        state = {
            'net': best_model_wts,
        }
        torch.save(state, save_path)
    else:
        early_stopping_counter += 1
        
    if early_stopping_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break
print('end')

# plt.figure(figsize=(30,15))
# plt.subplot(1,2,1)
# plt.plot(train_loss_all,"ro-",label = "Train loss")
# plt.legend()
# plt.xlabel("epoch")
# plt.ylabel("Loss")
# plt.title('Best test epoch: {0}'.format(best_epoch-1))
# plt.subplot(1,2,2)
# plt.plot(train_acc_all,"ro-",label = "Train acc")
# plt.plot(test_acc_all,"bs-",label = "Test acc")
# plt.xlabel("epoch")
# plt.ylabel("acc")
# plt.title('Best test Acc: {0}'.format(best_acc))
# plt.legend()
# plt.savefig("./total_loss.png")
# plt.show()

