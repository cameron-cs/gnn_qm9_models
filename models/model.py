import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool


# a single GCN layer
class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)  # initialize GCN convolution layer

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)  # perform graph convolution
        x = F.relu(x)  # A\apply ReLU activation
        return x


# a readout layer to aggregate node features
class ReadoutLayer(nn.Module):
    def __init__(self):
        super(ReadoutLayer, self).__init__()

    def forward(self, x, batch):
        x = global_mean_pool(x, batch)  # apply global mean pooling
        return x


# fully connected layers for final classification
class FullyConnectedLayer(nn.Module):
    def __init__(self, hidden_channels):
        super(FullyConnectedLayer, self).__init__()
        self.lin1 = nn.Linear(in_features=hidden_channels, out_features=64)  # linear layer 1
        self.lin2 = nn.Linear(in_features=64, out_features=1)  # linear layer 2

    def forward(self, x):
        x = F.relu(self.lin1(x))  # apply ReLU activation
        x = F.dropout(x, p=0.5, training=self.training)  # apply dropout
        x = self.lin2(x)  # final linear layer
        return x


class GraphClassificationModel(nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GraphClassificationModel, self).__init__()
        self.gcn1 = GCNLayer(num_features, hidden_channels)  # first GCN layer
        self.gcn2 = GCNLayer(hidden_channels, hidden_channels)  # second GCN layer
        self.gcn3 = GCNLayer(hidden_channels, hidden_channels)  # third GCN layer
        self.readout = ReadoutLayer()  # readout layer
        self.fc = FullyConnectedLayer(hidden_channels)  # fully connected layers

    def forward(self, x, edge_index, batch):
        x = self.gcn1(x, edge_index)  # apply first GCN layer
        x = self.gcn2(x, edge_index)  # apply second GCN layer
        x = self.gcn3(x, edge_index)  # apply third GCN layer
        x = self.readout(x, batch)  # apply readout layer
        x = self.fc(x)  # apply fully connected layers
        return x.view(-1)
