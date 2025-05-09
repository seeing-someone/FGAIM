import pickle

import numpy as np
import torch
from dgl import load_graphs


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def get_data(fold):
    filename = 'dataset/cross_data/fold-'+str(fold)+'/'

    with open(filename+'fingerprint_dict.pickle', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    # print(len(fingerprint_dict))
    MolCLR_train = torch.load(filename+'h_train.pth')
    graph_train, label_dict_train = load_graphs(filename+'dgl_train.bin')
    label_train = torch.squeeze(label_dict_train['graph_labels'])   # 在这里将他们两分开的
    maccs_train = torch.load(filename+'maccs_train.pth')
    morgan_train = torch.load(filename+'morgan_train.pth')
    # print(count(graph_train))
    print( label_train.shape)
    print(MolCLR_train.shape)
    print(maccs_train.shape)
    print(morgan_train.shape)
    print(len(graph_train))
    dataset_train = list(zip(graph_train, label_train, MolCLR_train, maccs_train, morgan_train))
    dataset_train = shuffle_dataset(list(dataset_train), 1234)

    MolCLR_test = torch.load(filename+'h_test.pth')
    graph_test, label_dict_test = load_graphs(filename+'dgl_test.bin')
    label_test = torch.squeeze(label_dict_test['graph_labels'])
    maccs_test = torch.load(filename + 'maccs_test.pth')
    morgan_test = torch.load(filename + 'morgan_test.pth')
    dataset_test = list(zip(graph_test, label_test, MolCLR_test, maccs_test, morgan_test))
    dataset_test = shuffle_dataset(list(dataset_test), 1234)

    return fingerprint_dict, dataset_train, dataset_test
