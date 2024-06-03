import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        return x


class ReadoutLayer(nn.Module):
    def __init__(self):
        super(ReadoutLayer, self).__init__()

    def forward(self, x, batch):
        x = global_mean_pool(x, batch)
        return x


class FullyConnectedLayer(nn.Module):
    def __init__(self, hidden_channels):
        super(FullyConnectedLayer, self).__init__()
        self.lin1 = nn.Linear(in_features=hidden_channels, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GraphClassificationModel(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GraphClassificationModel, self).__init__()
        self.gcn1 = GCNLayer(num_features, hidden_channels)
        self.gcn2 = GCNLayer(hidden_channels, hidden_channels)
        self.gcn3 = GCNLayer(hidden_channels, hidden_channels)
        self.readout = ReadoutLayer()
        self.fc = FullyConnectedLayer(hidden_channels)

    def forward(self, x, edge_index, batch):
        x = self.gcn1(x, edge_index)  # Correctly use self.gcn1
        x = self.gcn2(x, edge_index)  # Correctly use self.gcn2
        x = self.gcn3(x, edge_index)  # Correctly use self.gcn3
        x = self.readout(x, batch)  # Correctly use self.readout
        x = self.fc(x)  # Correctly use self.fc
        return x.view(-1)
