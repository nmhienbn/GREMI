import os
import logging
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pickle
import copy
import time
import torch
from model_GBM import *

# Env
from utils import *
import gc

if __name__ == "__main__":
    data_dir = str(sys.argv[1])
    model_save_dir = str(sys.argv[2])
    patience = int(sys.argv[3])
    tr_label = str(sys.argv[4])
    te_label = str(sys.argv[5])
    batch_size = int(sys.argv[6])

# DATA
# loaded_data = torch.load(data_dir)
#
data_tr = pd.concat(
    [
        pd.read_csv(f"{data_dir}/1_{tr_label}.csv"),
        pd.read_csv(f"{data_dir}/2_{tr_label}.csv"),
        pd.read_csv(f"{data_dir}/3_{tr_label}.csv"),
        pd.read_csv(f"{data_dir}/labels_{tr_label}.csv"),
    ],
    axis=1,
)
tr_omic = torch.tensor(data_tr.iloc[:, :-1].values, dtype=torch.float32)
tr_labels = torch.tensor(data_tr.iloc[:, -1].values, dtype=torch.long)
data_te = pd.concat(
    [
        pd.read_csv(f"{data_dir}/1_{te_label}.csv"),
        pd.read_csv(f"{data_dir}/2_{te_label}.csv"),
        pd.read_csv(f"{data_dir}/3_{te_label}.csv"),
        pd.read_csv(f"{data_dir}/labels_{te_label}.csv"),
    ],
    axis=1,
)
te_omic = torch.tensor(data_te.iloc[:, :-1].values, dtype=torch.float32)
te_labels = torch.tensor(data_te.iloc[:, -1].values, dtype=torch.long)
exp_adj1 = torch.tensor(
    pd.read_csv(f"{data_dir}/adj1.csv", header=0, index_col=0).values, dtype=torch.float32
)
exp_adj2 = torch.tensor(
    pd.read_csv(f"{data_dir}/adj2.csv", header=0, index_col=0).values, dtype=torch.float32
)
exp_adj3 = torch.tensor(
    pd.read_csv(f"{data_dir}/adj3.csv", header=0, index_col=0).values, dtype=torch.float32
)
# DATA LOADRE
tr_dataset = torch.utils.data.TensorDataset(tr_omic, tr_labels)
tr_data_loader = torch.utils.data.DataLoader(
    dataset=tr_dataset, batch_size=batch_size, shuffle=True
)
te_dataset = torch.utils.data.TensorDataset(te_omic, te_labels)
te_data_loader = torch.utils.data.DataLoader(
    dataset=te_dataset, batch_size=batch_size, shuffle=False
)


num_epochs = 3000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loss_function = nn.CrossEntropyLoss()
input_in_dim = [exp_adj1.shape[0], exp_adj2.shape[0], exp_adj3.shape[0]]
input_hidden_dim = [64]
network = Fusion(
    num_class=2,
    num_views=3,
    hidden_dim=input_hidden_dim,
    dropout=0.1,
    in_dim=input_in_dim,
    dim1=input_in_dim[0],
    dim2=input_in_dim[1],
    dim3=input_in_dim[2],
)
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
early_stopping_counter = 0
gc.collect()
torch.cuda.empty_cache()

for epoch in range(0, num_epochs):
    isPrint = epoch % 100 == 0
    # Print epoch
    if isPrint:
        print(" Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
    # Set current loss value
    network.train()
    current_loss = 0.0
    train_loss = 0.0
    train_corrects = 0
    train_num = 0

    for i, data in enumerate(tr_data_loader, 0):

        batch_x, targets = data
        batch_x1 = batch_x[:, : input_in_dim[0]].reshape(-1, input_in_dim[0], 1)
        batch_x2 = batch_x[:, input_in_dim[0] : -input_in_dim[2]].reshape(
            -1, input_in_dim[1], 1
        )
        batch_x3 = batch_x[:, -input_in_dim[2] :].reshape(-1, input_in_dim[2], 1)

        batch_x1 = batch_x1.to(torch.float32)
        batch_x2 = batch_x2.to(torch.float32)
        batch_x3 = batch_x3.to(torch.float32)
        targets = targets.long()
        batch_x1 = batch_x1.to(device, non_blocking=True)
        batch_x2 = batch_x2.to(device, non_blocking=True)
        batch_x3 = batch_x3.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        exp_adj1_device = exp_adj1.to(device, non_blocking=True)
        exp_adj2_device = exp_adj2.to(device, non_blocking=True)
        exp_adj3_device = exp_adj3.to(device, non_blocking=True)


        optimizer.zero_grad()
        (
            loss_fusion,
            tr_logits,
            gat_output1,
            gat_output2,
            gat_output3,
            output1,
            output2,
            output3,
        ) = network(batch_x1, batch_x2, batch_x3, exp_adj1_device, exp_adj2_device, exp_adj3_device, targets)
        tr_prob = F.softmax(tr_logits, dim=1)
        tr_pre_lab = torch.argmax(tr_prob, 1)

        loss = loss_fusion
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x1.size(0)
        train_corrects += torch.sum(tr_pre_lab == targets.data)
        train_num += batch_x1.size(0)
        
        # Free unused memory after each batch
        del batch_x, batch_x1, batch_x2, batch_x3, targets, loss_fusion, tr_logits
        del exp_adj1_device, exp_adj2_device, exp_adj3_device
        torch.cuda.empty_cache()
        gc.collect()
        
    # Evaluationfor this fold
    network.eval()
    test_loss = 0.0
    test_corrects = 0
    test_num = 0
    with torch.no_grad():
        for i, data in enumerate(te_data_loader, 0):
            batch_x, targets = data
            batch_x1 = batch_x[:, : input_in_dim[0]].reshape(-1, input_in_dim[0], 1)
            batch_x2 = batch_x[:, input_in_dim[0] : -input_in_dim[2]].reshape(
                -1, input_in_dim[1], 1
            )
            batch_x3 = batch_x[:, -input_in_dim[2] :].reshape(-1, input_in_dim[2], 1)
            batch_x1 = batch_x1.to(torch.float32)
            batch_x2 = batch_x2.to(torch.float32)
            batch_x3 = batch_x3.to(torch.float32)
            targets = targets.long()
            batch_x1 = batch_x1.to(device, non_blocking=True)
            batch_x2 = batch_x2.to(device, non_blocking=True)
            batch_x3 = batch_x3.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            exp_adj1_device = exp_adj1.to(device, non_blocking=True)
            exp_adj2_device = exp_adj2.to(device, non_blocking=True)
            exp_adj3_device = exp_adj3.to(device, non_blocking=True)

            te_logits = network.infer(
                batch_x1, batch_x2, batch_x3, exp_adj1_device, exp_adj2_device, exp_adj3_device
            )
            te_prob = F.softmax(te_logits, dim=1)
            te_pre_lab = torch.argmax(te_prob, 1)

            test_corrects += torch.sum(te_pre_lab == targets.data)
            test_num += batch_x1.size(0)
            
            del batch_x, batch_x1, batch_x2, batch_x3, targets, te_logits, te_pre_lab
            del exp_adj1_device, exp_adj2_device, exp_adj3_device
            torch.cuda.empty_cache()
            gc.collect()

    train_loss_all.append(train_loss / train_num)
    train_acc_all.append(train_corrects.double().item() / train_num)
    test_acc_all.append(test_corrects.double().item() / test_num)
    if isPrint:
        print(
            "{} Train Loss : {:.8f} Train ACC : {:.8f}".format(
                epoch, train_loss_all[-1], train_acc_all[-1]
            )
        )
        print("{}  Test ACC : {:.8f}".format(epoch, test_acc_all[-1]))

    if test_acc_all[-1] > best_acc:
        best_acc = test_acc_all[-1]
        best_epoch = epoch + 1
        best_model_wts = copy.deepcopy(network.state_dict())
        early_stopping_counter = 0
        # Saving the model
        save_path = model_save_dir
        state = {
            "net": best_model_wts,
        }
        torch.save(state, save_path)
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        print(f"Best test accuracy: {best_acc}")
        print(f"Best test epoch: {best_epoch}")
        break
    
print("end")

plt.figure(figsize=(30, 15))
plt.subplot(1, 2, 1)
plt.plot(train_loss_all, "ro-", label="Train loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("Best test epoch: {0}".format(best_epoch - 1))
plt.subplot(1, 2, 2)
plt.plot(train_acc_all, "ro-", label="Train acc")
plt.plot(test_acc_all, "bs-", label="Test acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.title("Best test Acc: {0}".format(best_acc))
plt.legend()
plt.savefig("/kaggle/working/total_loss.png")
plt.show()
