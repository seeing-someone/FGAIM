import copy
import os
import pandas as pd
import pickle
import sys
import time
import torch
from dgl import load_graphs
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric import data as DATA
from torch_geometric.data import Batch
import json
from graph_conversion import *
from metrics import *
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from torch.utils.data import Dataset
import torch


class GraphPairDataset(Dataset):
    def __init__(self, smile_list, prot_list, dta_graph, smiles_train, graph_train, label_train, MolCLR_train,
                 maccs_train, morgan_train):
        self.smile_list = smile_list
        self.prot_list = prot_list
        self.dta_graph = dta_graph
        self.smiles_train = smiles_train
        self.graph_train = graph_train
        self.label_train = label_train
        self.MolCLR_train = MolCLR_train
        self.maccs_train = maccs_train
        self.morgan_train = morgan_train

    def __len__(self):
        return len(self.smile_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        smile = self.smile_list[idx]
        prot = self.prot_list[idx]
        GCNData_Prot, GCNData_Smile = self.dta_graph[(prot, smile)]
        # print(prot+smile)
        # Find the index of the current smile in smiles_train
        train_idx = -1
        try:
            train_idx = self.smiles_train.index(smile)  # Find the index of smile in smiles_train
        except ValueError:
            print(f"Warning: SMILES {smile} not found in smiles_train.")

        # Retrieve corresponding information using the index
        GCNData_Smile.graph = self.graph_train[train_idx]  # Graph data for the current SMILE
        GCNData_Smile.h_CLR = self.MolCLR_train[train_idx]  # Other feature data
        GCNData_Smile.maccs = self.maccs_train[train_idx]
        GCNData_Smile.morgan = self.morgan_train[train_idx]

        return GCNData_Smile, GCNData_Prot


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB





# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_train_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    LOG_INTERVAL = 20
    for batch_idx, data in enumerate(train_loader):
        drug = data[0].to(device)
        prot = data[1].to(device)
        optimizer.zero_grad()
        # print(prot.edge_index)
        output,_ = model(drug, prot)
        loss = loss_fn(output, drug.y.long())
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(drug.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    print('Average loss: {:.4f}'.format(total_train_loss / (batch_idx + 1)))
    return total_train_loss / (batch_idx + 1)




def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_test_fea = torch.Tensor()  # 用于拼接所有的 test_fea
    loss_fn = torch.nn.CrossEntropyLoss()
    eval_loss = []
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            drug = data[0].to(device)
            prot = data[1].to(device)
            output,test_fea = model(drug, prot)
            loss = loss_fn(output,drug.y.long())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, drug.y.view(-1, 1).cpu()), 0)
            total_test_fea = torch.cat((total_test_fea, test_fea.cpu()), 0)  # 拼接 test_fea

            eval_loss.append(loss.item())
        eval_loss.append(loss.item())
    eval_loss = np.average(eval_loss)
    return total_labels.numpy().flatten(), total_preds.numpy(), eval_loss,total_test_fea



