#!/usr/bin/env python
# coding: utf-8
# In[ ]:
import math
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import dgl
import numpy as np
import torch
import torch.utils.data
from dgl import save_graphs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from sklearn.model_selection import KFold
from torch_geometric.data import Data as PYG_Data
from torch_geometric.loader import DataLoader

from ginet_molclr import GINet

from utils import makedirs

# Check if GPU is available
cuda_name = "cuda:0"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

# Features required for pre-trained models
ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


# =====================================================
def dump_dictionary(dictionary, file_name):
    """Save dictionary object"""
    with open(file_name, 'wb') as f:
        pickle.dump(dict(dictionary), f)


# ======================================================
def molToSmi(mol):
    if mol != None:
        smi = Chem.MolToSmiles(mol)
        if smi != '':
            return smi


# ======================================================
def smiToMol(smi):
    if type(smi) == str and smi != '':
        mol = Chem.MolFromSmiles(smi)
        return mol


# ======================================================
def standardizeSmi(smi, kekuleSmiles=False):
    """Check if the smile is a standard smile"""
    smi = molToSmi(smiToMol(smi))
    if kekuleSmiles:
        mol = smiToMol(smi)
        if mol:
            Chem.Kekulize(mol)
            smi_keku = Chem.MolToSmiles(mol, kekuleSmiles=True)
            if smi_keku != '':
                return smi_keku
    else:
        return smi


# ==============================================================
def create_ijbonddict(eType, eId):
    i_jbond_dict = defaultdict(lambda: [])
    eId_new = eId.transpose(1, 0)
    eType_id = np.where(eType)[1]
    for num, (i, j) in enumerate(eId_new):
        i_jbond_dict[i].append((j, eType_id[num]))
    return i_jbond_dict


# =========================================
def create_fingerprints(nType, i_jbond_dict, radius,smi_org, count):
    """Extract the r-radius subgraphs from a molecular graph using WeisfeilerLehman-like algorithm."""
    try:
        nType_id = torch.from_numpy(np.where(nType)[1])
        if (len(nType_id) == 1) or (radius == 0):
            fingerprints = [fingerprint_dict[a] for a in nType_id]
        else:
            vertices = np.array(nType_id)
            for _ in range(radius):
                fingerprints = []
                for i, j_bond in i_jbond_dict.items():
                    neighbors = [(vertices[j], bond) for j, bond in j_bond]
                    fingerprint = (vertices[i], tuple(sorted(neighbors)))
                    fingerprints.append(fingerprint_dict[fingerprint])
                vertices = fingerprints
        fingerprints = np.array(fingerprints)
        if len(fingerprints) < count:
            padding = np.ones(count - len(fingerprints), dtype=fingerprints.dtype)
            fingerprints = np.concatenate([fingerprints, padding])
        return fingerprints
    except Exception as e:
        print(f"{smi_org} An unexpected error occurred: {e}")
        return np.ones(count, dtype=int)


# =============================================
def get_morgan_fingerprint(mol, morgan_fp_N=200, radius=2):
    """Get morgan fingerprint"""
    nBits = morgan_fp_N
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return [int(b) for b in mfp.ToBitString()]


# =============================================
def get_maccs_fingerprint(mol):
    """Get maccs fingerprint"""
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    return [int(b) for b in fp.ToBitString()]


# =============================================
def get_PYG_DGL(dataset):
    """Obtain PYG graphs (input pre-trained model MolCLR) and DGL graphs (input MSGNN)"""
    graphs_PYG_DGL = []
    ans =1
    for i, graph in enumerate(dataset):
        # try:
            nType = graph.nf[:, :11]

            node_feat = torch.tensor(graph.nf, dtype=torch.float32)
            subgraph_feat = torch.tensor(graph.sf, dtype=torch.long)
            print(len(node_feat))
            print(len(subgraph_feat))
            print('^^^'*10)
            eId = torch.tensor(graph.eId)

            # Save PYG graph (as input for pre-trained model MolCLR)
            x_clr = graph.x_CLR.to(device)
            edge_attr_clr = graph.edge_attr_CLR.to(device)
            edges_clr = graph.index_CLR.type(torch.int64).to(device)
            y = torch.tensor(graph.yTrue).to(device)
            g_PYG = PYG_Data(x=x_clr, edge_index=edges_clr, edge_attr=edge_attr_clr, y=y)

            # Save DGL graph (as input for MSGNN)
            g_DGL = dgl.graph((eId[0], eId[1]), num_nodes=nType.shape[0])
            g_DGL.ndata['feature'] = node_feat
            g_DGL.ndata['subgraph'] = subgraph_feat
            print(ans)
            ans =ans +1
            graphs_PYG_DGL.append((g_PYG, g_DGL, graph.yTrue, graph.smi, graph.morgan, graph.maccs))
        # except:
        #     continue
    return graphs_PYG_DGL


# =============================================
def get_pretrained_embedding(model, graphs):
    """Using pre-trained model MolCLR"""
    graphs_MolCLR, graphs_MSGNN, labels, smiles, morgan, maccs = list(zip(*graphs))

    h_list, out_list, graph_list, y_list, smiles_list, morgan_list, maccs_list = [], [], [], [], [], [], []
    batch_size = 32
    loader = DataLoader(graphs_MolCLR, batch_size=batch_size, drop_last=True, shuffle=False)
    with torch.no_grad():
        for i, graph_batch in enumerate(loader):
            try:
                h, out = model(graph_batch.to(device))
            except:
                pass
            else:
                h = h.to('cpu')
                out = out.to('cpu')
                h_list.append(h)
                out_list.append(out)

                del h, out, graph_batch
                torch.cuda.empty_cache()

                for j in range(batch_size):
                    graph_list.append(graphs_MSGNN[i * batch_size + j])
                    y_list.append(labels[i * batch_size + j])
                    morgan_list.append(morgan[i * batch_size + j])
                    maccs_list.append(maccs[i * batch_size + j])
                    smiles_list.append(smiles[i * batch_size + j])
    return h_list, out_list, graph_list, y_list, smiles_list, morgan_list, maccs_list


# =============================================
class Graph():
    def __init__(self, smi, nf, ef, eId, yTrue, x, edge_attr, index, sf, morgan, maccs):
        self.smi = smi
        self.nf = nf
        self.ef = ef
        self.eId = eId
        self.yTrue = yTrue
        self.x_CLR = x  # Atomic features required for pre-trained MolCLR model
        self.edge_attr_CLR = edge_attr  # Bond features required for pre-trained MolCLR model
        self.index_CLR = index  # Edge indexes required for pre-trained MolCLR model
        self.sf = sf
        self.morgan = morgan
        self.maccs = maccs


# ==============================================================
class MolGraph():
    def __init__(self):
        # Atomic feature dictionary
        self.ATOM_FEATURE = {
            'AtomNumber': [6, 7, 8, 9, 15, 16, 17, 35, 53, 'Mask'],
            'Degree': [0, 1, 2, 3, 4, 5, 'Mask'],
            'NumH': [0, 1, 2, 3, 4, 'Mask'],
            'Hybridization': [HybridizationType.S,
                              HybridizationType.SP,
                              HybridizationType.SP2,
                              HybridizationType.SP3,
                              HybridizationType.SP3D,
                              HybridizationType.SP3D2,
                              HybridizationType.OTHER,
                              'Mask'],
            'Ring': [0, 3, 4, 5, 6, 7, 8, 'Mask'],
            'ImplicitValence': [0, 1, 2, 3, 4, 5, 'Mask'],
            'Aromatic': ['', 'Mask'],
            'FormalCharge': [''],
            'Mass': [''],
            'NumRadicalElectrons': ['']
        }
        # Bond feature dictionary
        self.BOND_FEATURE = {
            'bond': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
        }

        # Atomic feature index
        self.atomFeature = {}
        for name in self.ATOM_FEATURE:
            for e in self.ATOM_FEATURE[name]:
                self.atomFeature[name + str(e)] = len(self.atomFeature)
        # Bond feature index
        self.bondFeature = {}
        for name in self.BOND_FEATURE:
            for e in self.BOND_FEATURE[name]:
                self.bondFeature[name + str(e)] = len(self.bondFeature)
        # Number of atomic features
        self.atomFeatureNum = len(self.atomFeature)
        # Number of bond features
        self.bondFeatureNum = len(self.bondFeature)

    def get_graph(self, idx,data, aug_num=0, train=True):
        """ Obtain the original and augmented graphs """

        smi_org, property_indices = data.strip().split('\t')
        yTrue = property_indices.strip().split(',')

        smi = standardizeSmi(smi_org)
        if smi is None:
            print('standardizeSmi fail:  '+smi_org)
            # return None

        mol = smiToMol(smi)
        if mol is None:
            print('smiToMol fail:' +smi)
            # return None

        nf = self.get_node_feature(mol)
        # print(nf.shape)
        # print(nf.shape[0])
        # print(nf.shape[1])
        # print("***"*10)
        ef, eId = self.get_edge_feature(mol)

        x, edge_attr, index = self.get_pretrain_feature(mol)

        morgan_fingerprint = np.array(get_morgan_fingerprint(mol), 'int64')
        maccs_fingerprint = np.array(get_maccs_fingerprint(mol), 'int64')

        label = np.zeros((1, 11))
        for y in yTrue:
            label[0, int(y)] = 1

        i_jbond_dict = create_ijbonddict(ef, eId)

        sf = create_fingerprints(nf[:, :11], i_jbond_dict, radius,smi_org, nf.shape[0])
        print(len(sf))
        print('----'*10)
        # if(idx == 50):
        #     assert 1==0
        graph = Graph(smi_org, nf, ef, eId, label, x, edge_attr, index, sf, morgan_fingerprint, maccs_fingerprint)

        if False:
            graph_aug = self.get_mask_graph(graph, aug_num)
            return graph, graph_aug
        else:
            return graph



    def get_node_feature(self, mol):
        """Obtain the atomic features required for MSGNN"""
        nf = []
        for i, atom in enumerate(mol.GetAtoms()):
            nf_i = np.zeros(self.atomFeatureNum, dtype=np.float32)

            if atom.GetAtomicNum() in self.ATOM_FEATURE['AtomNumber']:
                nf_i[self.atomFeature['AtomNumber' + str(atom.GetAtomicNum())]] = 1.0

            if atom.GetTotalDegree() in self.ATOM_FEATURE['Degree']:
                nf_i[self.atomFeature['Degree' + str(atom.GetTotalDegree())]] = 1.0

            if atom.GetTotalNumHs() in self.ATOM_FEATURE['NumH']:
                nf_i[self.atomFeature['NumH' + str(atom.GetTotalNumHs())]] = 1.0

            if atom.GetHybridization() in self.ATOM_FEATURE['Hybridization']:
                nf_i[self.atomFeature['Hybridization' + str(atom.GetHybridization())]] = 1.0

            if atom.IsInRing():
                for ringSize in self.ATOM_FEATURE['Ring'][:-1]:
                    if atom.IsInRingSize(ringSize):
                        nf_i[self.atomFeature['Ring' + str(ringSize)]] = 1.0
                        break
            else:
                nf_i[self.atomFeature['Ring0']] = 1.0

            if atom.GetImplicitValence() in self.ATOM_FEATURE['ImplicitValence']:
                nf_i[self.atomFeature['ImplicitValence' + str(atom.GetImplicitValence())]] = 1.0

            if atom.GetIsAromatic():
                nf_i[self.atomFeature['Aromatic']] = 1.0

            nf_i[self.atomFeature['FormalCharge']] = atom.GetFormalCharge()

            nf_i[self.atomFeature['Mass']] = atom.GetMass() * 0.01

            nf_i[self.atomFeature['NumRadicalElectrons']] = atom.GetNumRadicalElectrons()
            nf.append(nf_i)
        return np.array(nf)

    def get_edge_feature(self, mol):
        """Obtain the bond features required for MSGNN"""
        eId_i, eId_j = [], []
        ef = []
        for bond in mol.GetBonds():
            ef_ij = np.zeros(self.bondFeatureNum, dtype=np.float32)
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = bond.GetBondType()
            eId_i += [start, end]
            eId_j += [end, start]
            ef_ij[self.bondFeature['bond' + str(bt)]] = 1.0
            ef.append(ef_ij)
            ef.append(ef_ij)

        ef = torch.tensor(np.array(ef), dtype=torch.long)
        eId = [eId_j, eId_i]

        return np.array(ef), np.array(eId)

    def get_pretrain_feature(self, mol):
        """Obtain the features required for the pre-trained model MolCLR"""
        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))  # GetAtomicNum获得原子序号
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))  # 手性
            atomic_number.append(atom.GetAtomicNum())
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            index = torch.tensor([row, col], dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        return x, edge_attr, index

    def get_mask_graph(self, graph, num):
        """obtain the augmented graphs"""
        mol = Chem.MolFromSmiles(graph.smi)
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        graph_aug = []
        for _ in range(num):
            graph_aug.append(self.mask_nodes_edges(graph, M, N))
        return graph_aug

    def mask_nodes_edges(self, graph, M, N):
        """Atom masking and bond deletion"""
        num_mask_nodes = max([1, math.floor(0.25 * N)])
        num_mask_edges = max([0, math.floor(0.25 * M)])

        mask_nodes = random.sample(list(range(N)), num_mask_nodes)

        mask_edges_1_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges_1 = [2 * i for i in mask_edges_1_single] + [2 * i + 1 for i in mask_edges_1_single]

        nf_new = deepcopy(graph.nf)
        for atom_idx in mask_nodes:
            nf_new[atom_idx, :] = np.zeros(graph.nf[
                                               atom_idx].shape)  # 连续值不需要用0再代替了，因为这已经全初始化为0了  'formalCharge'，'aromatic'，'mass'，'NumRadicalElectrons'
            nf_new[atom_idx, self.atomFeature['AtomNumberMask']] = 1.0
            nf_new[atom_idx, self.atomFeature['DegreeMask']] = 1.0
            nf_new[atom_idx, self.atomFeature['NumHMask']] = 1.0
            nf_new[atom_idx, self.atomFeature['HybridizationMask']] = 1.0
            nf_new[atom_idx, self.atomFeature['RingMask']] = 1.0
            nf_new[atom_idx, self.atomFeature['ImplicitValenceMask']] = 1.0
            nf_new[atom_idx, self.atomFeature['AromaticMask']] = 1.0

        edge_index_new = np.zeros((2, 2 * (M - num_mask_edges)), dtype=np.long)
        edge_attr_new = np.zeros((2 * (M - num_mask_edges), 4), dtype=np.long)
        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges_1:
                edge_index_new[:, count] = graph.eId[:, bond_idx]
                edge_attr_new[count, :] = graph.ef[bond_idx, :]
                count += 1
        graph_new = Graph(graph.smi, nf_new, edge_attr_new, edge_index_new, graph.yTrue, graph.x_CLR,
                          graph.edge_attr_CLR,
                          graph.index_CLR, graph.sf, graph.morgan, graph.maccs)
        return graph_new


# =============================================================================
def create_cross_data(ori_train, ori_test, fold, aug_num=0):
    """Create data for each fold in cross validation"""
    file_name = './cross_data/fold-' + str(fold) + '/'
    makedirs(file_name)

    train_dataset, test_dataset = [], []

    print(f'Fold-{fold}')

    # Train set data augmentation
    molGraph = MolGraph()
    # for data in ori_train:
    #     try:
    #         graph, graph_aug = molGraph.get_graph(data, aug_num, train=True)
    #         if graph is not None and (g is not None for g in graph_aug):
    #             train_dataset.append(graph)
    #             for i in range(aug_num):
    #                 train_dataset.append(graph_aug[i])
    #     except:
    #         continue
    print(f"Number of items before processing: {len(ori_test)}")  # 2177条
    # Test set data augmentation
    idx = 1
    for data in ori_test:
            graph = molGraph.get_graph(idx,data, train=False)
            if graph is not None:
                test_dataset.append(graph)
                idx = idx+1

    print("after finished:  " + str(len(test_dataset)))  # 2158条
    graphs_train = get_PYG_DGL(list(test_dataset))
    graphs_test = get_PYG_DGL(list(test_dataset))
    print("After get_PYG_DGL  " + str(len(graphs_test)))  # 2125条
    # Load pre-trained model MolCLR
    model = GINet(num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0, pool='mean')
    model.load_state_dict(torch.load('../graphCLR/pretrained_gin/checkpoints/model.pth', map_location=cuda_name))
    model.to(device)

    h_train, out_train, graph_train, y_train, smiles_train,morgan_train, maccs_train = get_pretrained_embedding(model, graphs_train)
    h_test, out_test, graph_test, y_test,  smiles_test,morgan_test, maccs_test = get_pretrained_embedding(model, graphs_test)
    print("Length of h_test:", len(h_test))  # 2112条
    print("Length of out_test:", len(out_test))
    print("Length of graph_test:", len(graph_test))
    print("Length of y_test:", len(y_test))
    print("Length of morgan_test:", len(morgan_test))
    print("Length of maccs_test:", len(maccs_test))
    print(f'The final saved data length is: \n Train set:{len(graph_train)} \t Test set:{len(graph_test)}')
    print(len(fingerprint_dict))
    dump_dictionary(fingerprint_dict, file_name + 'fingerprint_dict.pickle')

    # Save the embedding of the pre-trained model MolCLR
    torch.save(torch.cat(h_train), file_name + 'h_train.pth')
    torch.save(torch.cat(h_test), file_name + 'h_test.pth')

    # Save Morgan and MACCS fingerprints
    torch.save(torch.tensor(morgan_train), file_name + 'morgan_train.pth')
    torch.save(torch.tensor(maccs_train), file_name + 'maccs_train.pth')
    torch.save(torch.tensor(morgan_test), file_name + 'morgan_test.pth')
    torch.save(torch.tensor(maccs_test), file_name + 'maccs_test.pth')
    
    torch.save(smiles_train, file_name + 'smiles_train.pth')
    torch.save(smiles_test, file_name + 'smiles_test.pth')

    # Save the required molecular graphs for MSGNN
    y_list = {"graph_labels": torch.tensor(y_train)}
    save_graphs(file_name + 'dgl_train.bin', graph_train, y_list)

    y_list = {"graph_labels": torch.tensor(y_test)}
    save_graphs(file_name + 'dgl_test.bin', graph_test, y_list)


if __name__ == '__main__':
    radius = 2  # Subgraph radius
    aug_num = 10  # Data augmentation factor
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

    with open('work_4.txt', 'r') as f:
        ori_datas = f.read().strip().split('\n')
    print(f'The length of the dataset is: {len(ori_datas)}')

    # Ten fold cross validation
    # kf = KFold(n_splits=1, shuffle=True, random_state=1234)
    #
    # for i, (train_index, test_index) in enumerate(kf.split(ori_datas)):
    #     print('---------------------------------------------------')
    #     ori_train = [ori_datas[index] for index in train_index]
    #     ori_test = [ori_datas[index] for index in test_index]
    #
    #     create_cross_data(ori_train, ori_test, i+1, aug_num=aug_num)

    # 不进行数据分割，训练和测试均为 ori_datas 的全体数据
    print('---------------------------------------------------')
    ori_train = ori_datas  # 训练数据为全体数据
    ori_test = ori_datas  # 测试数据也为全体数据

    create_cross_data(ori_train, ori_test, 1, aug_num=aug_num)  # 第1次处理

