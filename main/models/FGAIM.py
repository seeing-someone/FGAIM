import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATConv
from torch_geometric.nn import TransformerConv
import time
from models.configs import get_model_defaults
from models.config import make_args
from dgl.nn.pytorch.glob import SumPooling, AvgPooling
import dgl
from models.GAT import (
    GAT
)

from models.GIN import (
    GIN
)


class ResidualBlock(torch.nn.Module):
    def __init__(self, outfeature):
        super(ResidualBlock, self).__init__()
        self.outfeature = outfeature
        self.sage = SAGEConv(outfeature, outfeature)
        self.ln = torch.nn.Linear(outfeature, outfeature, bias=False)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # 初始化SAGEConv的权重
        nn.init.xavier_uniform_(self.sage.lin_l.weight)

    def forward(self, x, edge_index):
        identity = x
        out = self.sage(x, edge_index)
        out = self.relu(out)
        out = self.ln(out)
        out += identity
        out = self.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(self.act1(self.fc1(x)))
        x = self.bn2(self.act2(self.fc2(x)))
        x = self.bn3(self.act3(self.fc3(x)))
        x = self.fc4(x)
        return x


class Net(nn.Module):
    def __init__(self, n_fingerprint, args, device):
        super(Net, self).__init__()

        self.args = args
        self.device = device
        # dimension of subgraph features: 256
        self.embed = nn.Embedding(n_fingerprint, self.args.subgraph_dim)

        self.feat_conv = GAT(
            in_dim=self.args.feature_dim,
            num_hidden=128,
            out_dim=256,
            num_layers=2,
            nhead=5,
            nhead_out=1,
            concat_out=True,
            activation='relu',
            feat_drop=0,
            attn_drop=0,
            negative_slope=0.1,
            residual=True,
            norm=None,
            encoding=None,
        )

        self.conv = GIN(
            in_dim=self.args.subgraph_dim,
            num_hidden=self.args.subgraph_dim,
            out_dim=self.args.subgraph_dim,
            num_layers=2,
            dropout=0,
            activation=None,
            residual=True,
            norm=None,
        )
        # args.MolCLR_dim: 512
        self.adapter_MolCLR = nn.Linear(self.args.MolCLR_dim, self.args.MolCLR_dim)

        self.adapter_fingerprint = nn.Linear(167 + 200, self.args.MolCLR_dim)

        self.pred1 = nn.Linear(self.args.MolCLR_dim, 128, bias=False)  # self.args.MolCLR_dim, 256, bias=False
        self.pred2 = nn.Linear(256, 128, bias=False)
        self.pred3 = nn.Linear(128, 11, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, graph, h_MolCLR, maccs, morgan):
        x_subgraph = self.embed(graph.ndata['subgraph'])
        x_feature = graph.ndata['feature']
        x_subgraph = self.conv(graph, x_subgraph)
        x_feature = self.feat_conv(graph, x_feature)

        result = torch.cat((x_subgraph, x_feature), dim=1)
        # result = x_feature

        h_MolCLR = self.adapter_MolCLR(h_MolCLR.to(self.device)).to(self.device)

        x_fingerprint = torch.cat((maccs, morgan), 1).type(torch.float32)
        x_fingerprint = self.adapter_fingerprint(x_fingerprint)

        y_molecules = h_MolCLR.to(self.device)
        y_molecules += x_fingerprint.to(self.device)

        return result, y_molecules


    def __call__(self, batched_data, h_CLR, maccs, morgan, train=True):
        out = self.forward(batched_data.to(self.device), h_CLR.to(self.device), maccs.to(self.device),
                           morgan.to(self.device))  # Float   morgan.to(self.device)).type(torch.float32)
        return out


# GCN model
# GCN based model
class FGAIM(torch.nn.Module):
    def __init__(self, num_features_xd, num_features_xt,
                 latent_dim=64, dropout=0.2, n_output=2, device='cpu', n_hidden=128, **kwargs):
        super(FGAIM, self).__init__()

        self.n_output = n_output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(0.5)
        self.device = device
        self.num_rblock = 4

        # SMILES graph branch
        self.conv1_xd = SAGEConv(num_features_xd, num_features_xd)
        self.conv2_xd = SAGEConv(num_features_xd, num_features_xd * 2)
        self.rblock_xd = ResidualBlock(num_features_xd * 2)
        self.fc_g1_d = torch.nn.Linear(num_features_xd * 2, 1024)
        self.fc_g2_d = torch.nn.Linear(1024, num_features_xt)
        self.fc_g3_d = torch.nn.Linear(num_features_xt, latent_dim * 2)  # latent_dim * 2

        # attention
        self.first_linear = torch.nn.Linear(num_features_xt, num_features_xt)
        self.second_linear = torch.nn.Linear(num_features_xt, 1)

        # protein graph branch
        self.conv1_xt = SAGEConv(num_features_xt, latent_dim)
        self.conv2_xt = SAGEConv(latent_dim, latent_dim * 2)
        self.rblock_xt = ResidualBlock(latent_dim * 2)
        self.fc_g1_t = torch.nn.Linear(latent_dim * 2, 1024)
        self.fc_g2_t = torch.nn.Linear(1024, latent_dim * 2)  # (1024, latent_dim * 2)

        self.fc1 = nn.Linear(4 * latent_dim, 1024)  # (4 * latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

        # drug
        args = make_args()
        self.net_drug = Net(15185, args, device).to(device)  # drugAI: train:6136 /train:dtiam: 15185
        self.prot_rnn = nn.LSTM(512, 128, 1)


    def forward(self, drug, prot):
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x2, edge_index2, batch2, prot_lens, edge_attr2 = prot.x, prot.edge_index, prot.batch, prot.prot_len, prot.edge_attr
        # drug branch
        # -------------------------------------------------------------------------------------
        batched_data, h_CLR, maccs, morgan = drug.graph, drug.h_CLR, drug.maccs, drug.morgan
        h_CLR_reshaped = h_CLR.reshape(-1, 512)
        maccs_reshaped = maccs.reshape(-1, 167)
        morgan_reshaped = morgan.reshape(-1, 200)
        batched_data = dgl.batch(batched_data)  # 将图对象组合成一个批次
        result, glo_fea = self.net_drug(batched_data, h_CLR_reshaped, maccs_reshaped, morgan_reshaped)

        # --------------------------------------------------------------------------------------
        # result = torch.cat((result, x), dim=1)
        x = self.conv1_xd(result, edge_index)  # x, edge_index
        x = self.relu(x)
        x = self.conv2_xd(x, edge_index)
        x = self.relu(x)
        for i in range(self.num_rblock):
            x = self.rblock_xd(x, edge_index)
        x = gmp(x, batch)  # global max pooling
        # flatten
        x = self.relu(self.fc_g1_d(x))
        x = self.dropout(x)
        x = self.fc_g2_d(x)
        x = self.dropout(x)
        x_changedim = self.relu(self.fc_g3_d(x))

        # ---------------------------------------------------------
        glo_fea, _ = self.prot_rnn(glo_fea)
        glo_fea = self.relu(glo_fea)
        # ---------------------------------------------------------

        # protein branch
        dense_node, bool_node = to_dense_batch(x2,
                                               batch2)  # (batch, num_node, num_feat) the num_node is the max node, and is padded
        cur_idx = -1
        cur_batch = 0
        # mask to remove drug node out of protein graph later
        mask = torch.ones(batch2.size(0), dtype=torch.bool)
        for size in prot_lens:
            batch_dense_node = dense_node[cur_batch]
            masked_batch_dense_node = batch_dense_node[bool_node[cur_batch]][:-1]

            node_att = F.tanh(self.first_linear(masked_batch_dense_node))
            node_att = self.dropout1(node_att)
            node_att = self.second_linear(node_att)
            node_att = self.dropout1(node_att)
            node_att = node_att.squeeze()
            node_att = F.softmax(node_att, 0)

            cur_idx += size + 1
            idx_target = (edge_index2[0] == cur_idx).nonzero()
            edge_attr2[idx_target.squeeze()] = node_att.squeeze()

            idx_target = (edge_index2[1] == cur_idx).nonzero()
            edge_attr2[idx_target.squeeze()] = node_att.squeeze()

            x2[cur_idx] = x[cur_batch]
            mask[cur_idx] = False
            cur_batch += 1
        # mask to get back drug node from protein graph later
        mask_drug = ~mask
        # protein feed forward

        x2 = self.conv1_xt(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = self.conv2_xt(x2, edge_index2)
        x2 = self.relu(x2)
        for i in range(self.num_rblock):
            x2 = self.rblock_xt(x2, edge_index2)

        x2_nodrug = x2[mask]
        batch2_nodrug = batch2[mask]
        drug_after = x2[mask_drug]
        # global max pooling
        x2 = gmp(x2_nodrug, batch2_nodrug)  # 128-dim
        # flatten
        x2 = self.relu(self.fc_g1_t(x2))
        x2 = self.dropout(x2)
        x2 = self.fc_g2_t(x2)
        x2 = self.dropout(x2)

        x = x_changedim.unsqueeze(2)
        glo_fea = glo_fea.unsqueeze(2)
        drug_after = drug_after.unsqueeze(2)

        # x = torch.cat((drug_after, x), 2)
        x = torch.cat((drug_after, x, glo_fea), 2)
        # x = torch.max_pool1d(x, 2, 1)
        x = torch.max_pool1d(x, 3, 1)  # torch.max_pool1d(x, 2, 1)
        x = x.squeeze(2)

        # concat
        xc = torch.cat((x, x2), 1)  # x ? xd
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out,xc

