import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SumPooling, AvgPooling

from models.GAT import (
    GAT
)

from models.GIN import (
    GIN
)


# Define a Neural Network class
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

        self.adapter_fingerprint = nn.Linear(167+200, self.args.MolCLR_dim)

        self.pred1 = nn.Linear(self.args.MolCLR_dim, 256, bias=False)
        self.pred2 = nn.Linear(256, 128, bias=False)
        self.pred3 = nn.Linear(128, 11, bias=False)

    def forward(self, graph, h_MolCLR, maccs, morgan):
        x_subgraph = self.embed(graph.ndata['subgraph'])
        x_feature = graph.ndata['feature']

        x_subgraph = self.conv(graph, x_subgraph)
        sumpool = SumPooling()
        x_subgraph = sumpool(graph, x_subgraph)  # torch.Size([1, 256])

        x_feature = self.feat_conv(graph, x_feature)
        avgpool3 = AvgPooling()
        x_feature = avgpool3(graph, x_feature) # torch.Size([1, 256])



        h_MolCLR = self.adapter_MolCLR(h_MolCLR.to(self.device)).to(self.device)

        x_fingerprint = torch.cat((maccs, morgan), 1).type(torch.float32)
        x_fingerprint = self.adapter_fingerprint(x_fingerprint)

        y_molecules = torch.cat((x_subgraph, x_feature), 1).type(torch.float32) #torch.Size([1, 512])

        y_molecules += h_MolCLR.to(self.device)

        y_molecules += x_fingerprint.to(self.device)


        z_properties = self.pred1(y_molecules)
        z_properties = F.relu(z_properties)
        z_properties = F.dropout(z_properties, training=self.training)
        z_properties = self.pred2(z_properties)
        z_properties = F.relu(z_properties)
        z_properties = self.pred3(z_properties)

        return z_properties

    def __call__(self, batched_data, labels, h_CLR, maccs, morgan, train=True):
        out = self.forward(batched_data.to(self.device), h_CLR.to(self.device), maccs.to(self.device),
                           morgan.to(self.device)).type(torch.double)

        if train:
            loss = F.binary_cross_entropy(torch.sigmoid(out), labels.to(self.device))
            return loss
        else:
            return out