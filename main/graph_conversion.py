import torch
from rdkit import Chem
import networkx as nx
import config
import numpy as np
import pickle
import json
import os


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# def atom_features(atom):
#     results = one_of_k_encoding_unk(atom.GetSymbol(),
#                                     ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
#                                      'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
#                                      'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
#                                      'Unknown']) + \
#               one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
#               one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
#               [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
#               one_of_k_encoding_unk(atom.GetHybridization(),
#                                     [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#                                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#                                      Chem.rdchem.HybridizationType.SP3D2]) + \
#               [atom.GetIsAromatic()]
#     return np.array(results)

def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    dic['X'] = (dic['X'] - min_value) / interval
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

res_dict = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K',
            'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W',
            'TYR': 'Y'}

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
        if (pro_seq[i] == 'X'):
            print(residue_features(pro_seq[i]))
    return np.concatenate((pro_hot, pro_property), axis=1)


# def smile_to_graph(smile):
#     mol = Chem.MolFromSmiles(smile)
#
#     c_size = mol.GetNumAtoms()
#
#     features = []
#     for atom in mol.GetAtoms():
#         feature = atom_features(atom)
#         features.append(feature / sum(feature))
#
#     edges = []
#     for bond in mol.GetBonds():
#         edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
#     g = nx.Graph(edges).to_directed()
#     edge_index = []
#     for e1, e2 in g.edges:
#         edge_index.append([e1, e2])
#
#     return c_size, features, edge_index

def smile_to_graph(dataset, smile, ligands_id):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    return c_size, features, edge_index


def aa_features(aa):
    results = one_of_k_encoding(aa,
                                ['A', 'N', 'C', 'Q', 'H', 'L', 'M', 'P', 'T', 'Y', 'R', 'D', 'E', 'G', 'I', 'K', 'F',
                                 'S', 'W', 'V', 'X', 'U'])
    return np.asarray(results, dtype=float)


def aa_sas_feature(target, dataset='davis'):
    feature = []
    file = 'data/' + dataset + '/profile/' + target + '_PROP/' + target + '.acc'
    for line in open(file):
        if line[0] == '#':
            continue
        cols = line.strip().split()
        if len(cols) == 6:
            res_sas = []
            res_sas.append(cols[-3])
            res_sas.append(cols[-2])
            res_sas.append(cols[-1])
            feature.append(np.asarray(res_sas, dtype=float))
    return np.asarray(feature)


def aa_ss_feature(target, dataset='davis'):
    feature = []
    file = 'data/' + dataset + '/profile/' + target + '_PROP/' + target + '.ss8'
    for line in open(file):
        cols = line.strip().split()
        if len(cols) == 11:
            res_sas = []
            res_sas.append(cols[-8])
            res_sas.append(cols[-7])
            res_sas.append(cols[-6])
            res_sas.append(cols[-5])
            res_sas.append(cols[-4])
            res_sas.append(cols[-3])
            res_sas.append(cols[-2])
            res_sas.append(cols[-1])
            feature.append(np.asarray(res_sas, dtype=float))
    return np.asarray(feature)


def prot_to_graph(seq, prot_contactmap, prot_target, dataset):
    c_size = len(seq)
    eds_seq = []
    if config.is_seq_in_graph:
        for i in range(c_size - 1):
            eds_seq.append([i, i + 1])
        eds_seq = np.array(eds_seq)
    eds_contact = []
    if config.is_con_in_graph:
        eds_contact = np.array(np.argwhere(prot_contactmap >= 0.5))

    # add an reserved extra node for drug node
    eds_d = []
    for i in range(c_size):
        eds_d.append([i, c_size])

    eds_d = np.array(eds_d)
    if config.is_seq_in_graph and config.is_con_in_graph:
        eds = np.concatenate((eds_seq, eds_contact, eds_d))
    elif config.is_con_in_graph:
        eds = np.concatenate((eds_contact, eds_d))
    else:
        eds = np.concatenate((eds_seq, eds_d))

    edges = [tuple(i) for i in eds]
    g = nx.Graph(edges).to_directed()
    features = []
    ss_feat = []
    sas_feat = []
    aved_prot = []
    if config.is_profile_in_graph:
        # pkl_file_dssp = os.path.join(dataset + 'dssp_feature', f'{prot_target}.pkl')
        # with open(pkl_file_dssp, 'rb') as f:
        #     saved_dssp_features = pickle.load(f)
        # pkl_file_atom = os.path.join(dataset + 'Atom_Feature', f'{prot_target}.pkl')
        # with open(pkl_file_atom, 'rb') as f:
        #     saved_atom_features = pickle.load(f)


        # data=torch.load(dataset + 'tape_feature', f'{prot_target}')
        # saved_tape = data['feature']

        # pkl_file_tape = os.path.join(dataset + 'tape', f'{prot_target}')
        # with open(pkl_file_tape, 'rb') as f:
        #     data= torch.load(f)
        #     saved_tape = data['feature']

        pt_file_prot = os.path.join(dataset + 'Prot embedding', f'{prot_target}.pt')
        # 使用 torch.load() 来读取 .pt 文件
        with open(pt_file_prot, 'rb') as f:
            saved_prot = torch.load(f)
        # node_features = np.concatenate((node_features, saved_atom_features), axis=1)
        # node_features = np.concatenate((node_features, saved_dssp_features), axis=1)
        # ss_feat = aa_ss_feature(prot_target, dataset)
        # sas_feat = aa_sas_feature(prot_target, dataset)
        # ss_feat = saved_dssp_features
        # sas_feat = saved_atom_features
        protTrans_feat = saved_prot
        # esm_tape_feat = saved_tape

        # if ss_feat.shape[0] < protTrans_feat.shape[0]:
        #     print(ss_feat.shape)
        #     print(sas_feat.shape)
        #     print(protTrans_feat.shape)
        #     pad_rows = protTrans_feat.shape[0] - ss_feat.shape[0]
        #     ss_feat = np.vstack([ss_feat, np.tile(ss_feat[-1:], (pad_rows, 1))])  # 从最后一行复制补齐

        # if ss_feat.shape[0] > protTrans_feat.shape[0]:
        #     print(prot_target)
        #     print("*****" * 10)
        #     print(ss_feat.shape)
        #     print(sas_feat.shape)
        #     print(protTrans_feat.shape)
        #     pad_rows = protTrans_feat.shape[0] - ss_feat.shape[0]
        #     ss_feat = np.vstack([ss_feat, np.tile(ss_feat[-1:], (pad_rows, 1))])  # 从最后一行复制补齐

        # 判断 sas_feat 行数是否小于 protTrans_feat 行数
        # if sas_feat.shape[0] < protTrans_feat.shape[0]:
        #     pad_rows = protTrans_feat.shape[0] - sas_feat.shape[0]
        #     sas_feat = np.vstack([sas_feat, np.tile(sas_feat[-1:], (pad_rows, 1))])  # 从最后一行复制补齐

    new_seq3 = ''.join([char if char in pro_res_table else 'X' for char in seq])
    node_features = seq_feature(new_seq3)

    # sequence_output = saved_tape
    # sequence_output = np.load('data/davis/emb/' + prot_target + '.npz', allow_pickle=True)
    # sequence_output = sequence_output[prot_target].reshape(-1, 1)[0][0]['seq'][1:-1, :]
    # sequence_output = sequence_output.reshape(sequence_output.shape[0], sequence_output.shape[1])
    for i in range(c_size):
        if config.is_profile_in_graph:
            if config.is_emb_in_graph:
                # aa_feat = np.concatenate((np.asarray(esm_tape_feat[i], dtype=float),))
                aa_feat = np.concatenate((np.asarray(protTrans_feat[i], dtype=float),))
                # aa_feat = np.concatenate((np.asarray(protTrans_feat[i], dtype=float),ss_feat[i], sas_feat[i],))
                # print(f'sequence_output[i]  {len(sequence_output[i])}, dtype=float), ss_feat[i] {len(ss_feat[i])}, sas_feat[i] {len(sas_feat[i])}')
                # aa_feat = np.concatenate((np.asarray(sequence_output[i], dtype=float), ss_feat[i], sas_feat[i]))
                # aa_feat = np.concatenate(( ss_feat[i], sas_feat[i]))
                # aa_feat = np.concatenate(( node_features[i],))
            else:
                aa_feat = np.concatenate((aa_features(seq[i]), ss_feat[i], sas_feat[i]))
        else:
            if config.is_emb_in_graph:
                aa_feat = np.asarray(sequence_output[i], dtype=float)
            else:
                aa_feat = aa_features(seq[i])
        features.append(aa_feat)
    # place holder feature vector for drug
    place_holder = np.zeros(features[0].shape, dtype=float)
    features.append(place_holder)

    np_array = np.array(features)
    print(np_array.shape)  # 输出形状应为 (N, 1024)

    edge_index = []
    edge_weight = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        # if e1 == c_size or e2 == c_size:
        #     edge_weight.append(0.5)
        # else:
        edge_weight.append(1.0)
    return c_size, features, edge_index, edge_weight