import numpy as np
import pandas as pd
import torch

data_tr = pd.read_csv('./ucec_train.csv', header=None).iloc[:,1:]
tr_omic = torch.tensor(data_tr.iloc[:,:-1].values, dtype=torch.float32)
tr_labels = torch.tensor(data_tr.iloc[:,-1].values, dtype=torch.long)
data_te = pd.read_csv('./ucec_test.csv', header=None).iloc[:,1:]
te_omic = torch.tensor(data_te.iloc[:,:-1].values, dtype=torch.float32)
te_labels = torch.tensor(data_te.iloc[:,-1].values, dtype=torch.long)
exp_adj1 = torch.tensor(pd.read_csv('./adj1.csv', header=0, index_col=0).values, dtype=torch.long)
exp_adj2 = torch.tensor(pd.read_csv('./adj2.csv', header=0, index_col=0).values, dtype=torch.long)
exp_adj3 = torch.tensor(pd.read_csv('./adj3.csv', header=0, index_col=0).values, dtype=torch.long)

data = {'data_tr': data_tr, 'tr_omic': tr_omic, 'tr_labels': tr_labels, 'data_te': data_te, 'te_omic': te_omic, 'te_labels': te_labels, 'exp_adj1': exp_adj1, 'exp_adj2': exp_adj2, 'exp_adj3': exp_adj3}

for key, value in data.items():
    print(key, value.shape)

torch.save(data, 'data.pt')