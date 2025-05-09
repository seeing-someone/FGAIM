import copy
import os
import pandas as pd
import pickle
import sys
import time
import torch
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
from utils import GraphPairDataset
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import *

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
from dgl import load_graphs


# dataset = config.dataset


def create_dataset_for_train(dataset):
    proteins = json.load(open(dataset + 'proteins.txt'), object_pairs_hook=OrderedDict)
    pdbs_seqs = []
    pdbs = []
    target_key = pdbs

    for t in proteins.keys():
        pdbs_seqs.append(proteins[t])
        pdbs.append(t)

    dta_graph = {}
    print('Pre-processing protein')
    print('Pre-processing...')
    saved_prot_graph = {}
    if os.path.isfile(dataset + 'saved_graph/' + 'saved_prot_graph.pickle'):
        print("Load pre-processed file for protein graph")
        with open(dataset + 'saved_graph/' + 'saved_prot_graph.pickle', 'rb') as handle:
            saved_prot_graph = pickle.load(handle)
    else:
        for target, seq in tqdm(set(zip(pdbs, pdbs_seqs)), desc="Processing targets"):
            contactmap = np.load(dataset + 'total_contact_map/' + target + '.npy')
            c_size, features, edge_index, edge_weight = prot_to_graph(seq, contactmap, target, dataset)
            features_np = np.array(features, dtype=np.float32)
            x = torch.tensor(features_np, dtype=torch.float32).to(device)  # 转换为 float 类型
            g = DATA.Data(
                x=x,
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                edge_attr=torch.FloatTensor(edge_weight),
                prot_len=c_size,
            )
            saved_prot_graph[target] = g
        with open(dataset + 'saved_graph/' + 'saved_prot_graph.pickle', 'wb') as handle:
            pickle.dump(saved_prot_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ligands = json.load(open(dataset + 'compounds.txt'), object_pairs_hook=OrderedDict)
    compound_iso_smiles = []
    smiles_to_drugid = {}
    for d in ligands.keys():
        # lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        compound_iso_smiles.append(ligands[d])
        smiles_to_drugid[ligands[d]] = d

    saved_drug_graph = {}
    if os.path.isfile(dataset + 'saved_graph/' + 'saved_drug_graph.pickle'):
        print("Load pre-processed file for drug graph")
        with open(dataset + 'saved_graph/' + 'saved_drug_graph.pickle', 'rb') as handle:
            saved_drug_graph = pickle.load(handle)
    else:
        for smiles in tqdm(compound_iso_smiles, desc="Processing SMILES"):
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
            ligands_id = smiles_to_drugid[smiles]
            c_size2, features2, edge_index2 = smile_to_graph(dataset, lg, ligands_id)
            features2_np = np.array(features2, dtype=np.float32)
            x = torch.tensor(features2_np, dtype=torch.float32).to(device)
            g2 = DATA.Data(
                x=torch.Tensor(features2),
                edge_index=torch.LongTensor(edge_index2).transpose(1, 0),
            )
            saved_drug_graph[smiles] = g2
        with open(dataset + 'saved_graph/' + 'saved_drug_graph.pickle', 'wb') as handle:
            pickle.dump(saved_drug_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Main program: iterate over different datasets  and encoding types:
    print("^^^^^" * 10)
    train_val_ratio = 0.9
    seed = 0
    affinity = pickle.load(open(dataset + 'Y', 'rb'), encoding='latin1')
    raw_fold = eval(open(dataset + 'valid_entries.txt', 'r').read())

    # Pick out test entries
    rows, cols = np.where(np.isnan(affinity) == False)
    rows, cols = rows[raw_fold], cols[raw_fold]

    test_fold_entries = {'compound_iso_smiles': [], 'target_key': [], 'affinity': []}
    for pair_ind in range(len(rows)):
        test_fold_entries['compound_iso_smiles'].append(compound_iso_smiles[rows[pair_ind]])
        test_fold_entries['target_key'].append(target_key[cols[pair_ind]])
        test_fold_entries['affinity'].append(affinity[rows[pair_ind], cols[pair_ind]])

    test_fold = pd.DataFrame(test_fold_entries)
    test_drugs = np.asarray(test_fold['compound_iso_smiles'])
    test_prot_keys = np.asarray(test_fold['target_key'])
    test_Y = np.asarray(test_fold['affinity'])

    dta_graph = {}
    num_feat_xp = 0
    num_feat_xd = 0

    # Loop through test data to build the graph
    for target, smile, label in zip(test_prot_keys, test_drugs, test_Y):
        g = saved_prot_graph[target]
        g2 = saved_drug_graph[smile]

        # Ensure features are available
        num_feat_xp = g.x.size(1)
        num_feat_xd = g2.x.size(1)

        # Move graphs to the device
        g = g.to(device)
        g2 = g2.to(device)

        # Assign labels
        g.y = torch.LongTensor([label]).to(device)
        g2.y = torch.LongTensor([label]).to(device)

        # Store in dta_graph
        # dta_graph[(target, smile)] = [g, g2]
        dta_graph[(target, smile)] = [copy.deepcopy(g), copy.deepcopy(g2)]

    ptr = int(train_val_ratio * len(raw_fold))
    train_valid = [raw_fold[:ptr], raw_fold[ptr:]]

    # load affinity matrix...
    print('load affinity matrix...')
    train_fold = train_valid[0]
    valid_fold = train_valid[1]
    print('train entries:', len(train_fold))
    print('valid entries:', len(valid_fold))
    stime = time.time()
    print('load train data...')
    rows, cols = np.where(np.isnan(affinity) == False)
    trows, tcols = rows[train_fold], cols[train_fold]  # trows和tcols将包含对应于affinity数组中非NaN元素且被train_fold选中的行和列索引。
    train_fold_entries = {'compound_iso_smiles': [], 'target_key': [], 'affinity': []}
    for pair_ind in range(len(trows)):
        train_fold_entries['compound_iso_smiles'] += [compound_iso_smiles[trows[pair_ind]]]
        train_fold_entries['target_key'] += [target_key[tcols[pair_ind]]]
        train_fold_entries['affinity'] += [affinity[trows[pair_ind], tcols[pair_ind]]]
    print('load valid data...')
    trows, tcols = rows[valid_fold], cols[valid_fold]
    valid_fold_entries = {'compound_iso_smiles': [], 'target_key': [], 'affinity': []}
    for pair_ind in range(len(trows)):
        valid_fold_entries['compound_iso_smiles'] += [compound_iso_smiles[trows[pair_ind]]]
        valid_fold_entries['target_key'] += [target_key[tcols[pair_ind]]]
        valid_fold_entries['affinity'] += [affinity[trows[pair_ind], tcols[pair_ind]]]
    print('done time consuming:', time.time() - stime)

    df_train_fold = pd.DataFrame(train_fold_entries)
    train_drugs, train_prot_keys, train_Y = list(df_train_fold['compound_iso_smiles']), list(
        df_train_fold['target_key']), list(df_train_fold['affinity'])
    train_drugs, train_prot_keys, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_Y)

    df_valid_fold = pd.DataFrame(valid_fold_entries)
    valid_drugs, valid_prots_keys, valid_Y = list(df_valid_fold['compound_iso_smiles']), list(
        df_valid_fold['target_key']), list(df_valid_fold['affinity'])
    valid_drugs, valid_prots_keys, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prots_keys), np.asarray(
        valid_Y)

    filename = '/public/home/tangyi/train_dtiam/fold-1/'
    with open(filename + 'fingerprint_dict.pickle', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    MolCLR_train = torch.load(filename + 'h_train.pth')
    graph_train, label_dict_train = load_graphs(filename + 'dgl_train.bin')
    label_train = torch.squeeze(label_dict_train['graph_labels'])
    maccs_train = torch.load(filename + 'maccs_train.pth')
    morgan_train = torch.load(filename + 'morgan_train.pth')
    smiles_train = torch.load(filename + 'smiles_train.pth')

    # make data PyTorch Geometric ready
    train_data = GraphPairDataset(smile_list=train_drugs, dta_graph=dta_graph, prot_list=train_prot_keys,
                                  smiles_train=smiles_train, graph_train=graph_train, label_train=label_train,
                                  MolCLR_train=MolCLR_train, maccs_train=maccs_train, morgan_train=morgan_train)
    valid_data = GraphPairDataset(smile_list=valid_drugs, dta_graph=dta_graph, prot_list=valid_prots_keys,
                                  smiles_train=smiles_train, graph_train=graph_train, label_train=label_train,
                                  MolCLR_train=MolCLR_train, maccs_train=maccs_train, morgan_train=morgan_train)

    # train_data = GraphPairDataset(smile_list=train_drugs, dta_graph=dta_graph, prot_list=train_prot_keys)
    # valid_data = GraphPairDataset(smile_list=valid_drugs, dta_graph=dta_graph, prot_list=valid_prots_keys)
    # test_data = GraphPairDataset(smile_list=test_drugs, dta_graph=dta_graph, prot_list=test_prots)
    # make data PyTorch mini-batch processing ready
    return train_data, valid_data, num_feat_xp, num_feat_xd


def create_dataset_for_test(dataset):
    proteins = json.load(open(dataset + 'proteins.txt'), object_pairs_hook=OrderedDict)
    pdbs_seqs = []
    pdbs = []
    target_key = pdbs

    for t in proteins.keys():
        pdbs_seqs.append(proteins[t])
        pdbs.append(t)

    dta_graph = {}
    print('Pre-processing protein')
    print('Pre-processing...')
    saved_prot_graph = {}
    if os.path.isfile(dataset + 'saved_graph/' + 'saved_prot_graph.pickle'):
        print("Load pre-processed file for protein graph")
        with open(dataset + 'saved_graph/' + 'saved_prot_graph.pickle', 'rb') as handle:
            saved_prot_graph = pickle.load(handle)
    else:
        for target, seq in tqdm(set(zip(pdbs, pdbs_seqs)), desc="Processing targets"):
            contactmap = np.load(dataset + 'total_contact_map/' + target + '.npy')
            c_size, features, edge_index, edge_weight = prot_to_graph(seq, contactmap, target, dataset)
            features_np = np.array(features, dtype=np.float32)
            x = torch.tensor(features_np, dtype=torch.float32).to(device)  # 转换为 float 类型
            g = DATA.Data(
                x=x,
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                edge_attr=torch.FloatTensor(edge_weight),
                prot_len=c_size,
            )
            saved_prot_graph[target] = g
        with open(dataset + 'saved_graph/' + 'saved_prot_graph.pickle', 'wb') as handle:
            pickle.dump(saved_prot_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ligands = json.load(open(dataset + 'compounds.txt'), object_pairs_hook=OrderedDict)
    compound_iso_smiles = []
    smiles_to_drugid = {}
    for d in ligands.keys():
        # lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        compound_iso_smiles.append(ligands[d])
        smiles_to_drugid[ligands[d]] = d

    saved_drug_graph = {}
    if os.path.isfile(dataset + 'saved_graph/' + 'saved_drug_graph.pickle'):
        print("Load pre-processed file for drug graph")
        with open(dataset + 'saved_graph/' + 'saved_drug_graph.pickle', 'rb') as handle:
            saved_drug_graph = pickle.load(handle)
    else:
        for smiles in tqdm(compound_iso_smiles, desc="Processing SMILES"):
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
            ligands_id = smiles_to_drugid[smiles]
            c_size2, features2, edge_index2 = smile_to_graph(dataset, lg, ligands_id)
            features2_np = np.array(features2, dtype=np.float32)
            x = torch.tensor(features2_np, dtype=torch.float32).to(device)
            g2 = DATA.Data(
                x=torch.Tensor(features2),
                edge_index=torch.LongTensor(edge_index2).transpose(1, 0),
            )
            saved_drug_graph[smiles] = g2
        with open(dataset + 'saved_graph/' + 'saved_drug_graph.pickle', 'wb') as handle:
            pickle.dump(saved_drug_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Main program: iterate over different datasets  and encoding types:
    print("^^^^^" * 10)
    train_val_ratio = 0.9
    seed = 0
    affinity = pickle.load(open(dataset + 'Y', 'rb'), encoding='latin1')
    test_fold = eval(open(dataset + 'valid_entries.txt', 'r').read())

    # load affinity matrix...
    print('load affinity matrix...')
    # pick out test entries
    rows, cols = np.where(np.isnan(affinity) == False)
    rows, cols = rows[test_fold], cols[test_fold]
    test_fold_entries = {'compound_iso_smiles': [], 'target_key': [], 'affinity': []}
    for pair_ind in range(len(rows)):
        test_fold_entries['compound_iso_smiles'] += [compound_iso_smiles[rows[pair_ind]]]
        test_fold_entries['target_key'] += [target_key[cols[pair_ind]]]
        test_fold_entries['affinity'] += [affinity[rows[pair_ind], cols[pair_ind]]]

    test_fold = pd.DataFrame(test_fold_entries)
    test_drugs = np.asarray(test_fold['compound_iso_smiles'])
    test_prot_keys = np.asarray(test_fold['target_key'])
    test_Y = np.asarray(test_fold['affinity'])

    dta_graph = {}
    num_feat_xp = 0
    num_feat_xd = 0

    # Loop through test data to build the graph
    for target, smile, label in zip(test_prot_keys, test_drugs, test_Y):
        g = saved_prot_graph[target]
        g2 = saved_drug_graph[smile]

        # Ensure features are available
        num_feat_xp = g.x.size(1)
        num_feat_xd = g2.x.size(1)

        # Move graphs to the device
        g = g.to(device)
        g2 = g2.to(device)

        # Assign labels
        g.y = torch.LongTensor([label]).to(device)
        g2.y = torch.LongTensor([label]).to(device)

        # Store in dta_graph
        # dta_graph[(target, smile)] = [g, g2]
        dta_graph[(target, smile)] = [copy.deepcopy(g), copy.deepcopy(g2)]

    test_fold = pd.DataFrame(test_fold_entries)
    test_drugs, test_prot_keys, test_Y = np.asarray(list(test_fold['compound_iso_smiles'])), np.asarray(
        list(test_fold['target_key'])), np.asarray(list(test_fold['affinity']))

    # for target, smile, label in zip(test_prot_keys, test_drugs, test_Y):
    #     t1, t2 = dta_graph[(target, smile)]

    # make data PyTorch Geometric ready
    filename = '/public/home/tangyi/tangyi/MSGNN-main（20241029）/MSGNN-main/dataset/cross_data/fold-1/'
    with open(filename + 'fingerprint_dict.pickle', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    MolCLR_train = torch.load(filename + 'h_train.pth')
    graph_train, label_dict_train = load_graphs(filename + 'dgl_train.bin')
    label_train = torch.squeeze(label_dict_train['graph_labels'])
    maccs_train = torch.load(filename + 'maccs_train.pth')
    morgan_train = torch.load(filename + 'morgan_train.pth')
    smiles_train = torch.load(filename + 'smiles_train.pth')

    test_data = GraphPairDataset(smile_list=test_drugs, dta_graph=dta_graph, prot_list=test_prot_keys,
                                 smiles_train=smiles_train, graph_train=graph_train, label_train=label_train,
                                 MolCLR_train=MolCLR_train, maccs_train=maccs_train, morgan_train=morgan_train)

    # test_data = GraphPairDataset(smile_list=test_drugs, dta_graph=dta_graph, prot_list=test_prot_keys)
    # make data PyTorch mini-batch processing ready
    return test_data, num_feat_xp, num_feat_xd

