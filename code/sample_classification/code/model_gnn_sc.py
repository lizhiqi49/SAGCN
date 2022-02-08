import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold
import torch_geometric.nn as gnn
from torch_geometric.data import Data


if torch.cuda.is_available():
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.cuda._initialized = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_feature = 250
num_class = 12
node_feature_len = 8


MAX_DIFF = 0.01
class MyDataset(Dataset):

    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        x_sample = self.Data[index].to(device)
        label = self.Label[index].to(device)
        x_graph = self.Data[index].numpy().repeat(node_feature_len).reshape(-1, node_feature_len)
        x_graph = torch.tensor(x_graph, dtype=torch.float).to(device)
        return x_sample, label, x_graph


"""
X: (6501, 250)
y: (6501, 1)
"""
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # sample feature part
        self.linear1 = nn.Sequential(
            nn.Linear(num_feature, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # interaction net part
        self.gconv1 = gnn.GCNConv(node_feature_len, node_feature_len)
        self.bn1 = nn.BatchNorm2d(1)
        self.activate1 = nn.ReLU()
        self.dp1 = nn.Dropout(0.1)

        self.gconv2 = gnn.GCNConv(node_feature_len, node_feature_len)
        self.bn2 = nn.BatchNorm2d(1)
        self.activate2 = nn.ReLU()
        self.dp2 = nn.Dropout(0.1)

        #self.top_k_pooling = gnn.TopKPooling(node_feature_len, 0.1)

        # after concatenate
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Conv1d(64, 128, 250)
        self.fc1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.output = nn.Linear(64, 12)

    def forward(self, x_sample, x_graph):
        #print(x_graph.shape)
        x_sample.to(device)
        x_graph.to(device)
        edge_index = self.get_edge_index(x_graph).to(device)
        # sample part
        x_sample = self.linear1(x_sample)

        # interaction net part
        batch_size, num_node, _ = x_graph.dim()
        x_graph = x_graph.view(-1, node_feature_len)
        x_graph = self.gconv1(x_graph, edge_index)
        x_graph = x_graph.view(batch_size, 1, num_node, node_feature_len)
        x_graph = self.dp1(self.activate1(self.bn1(x_graph)))
        """
        x_graph = x_graph.view(-1 ,node_feature_len)
        x_graph = self.gconv2(x_graph, edge_index)
        x_graph = x_graph.view(batch_size, 1, num_node, node_feature_len)
        x_graph = self.dp2(self.activate2(self.bn2(x_graph)))
        """
        x_graph = x_graph.view(batch_size, num_node, node_feature_len)
        x_graph = torch.mean(x_graph, dim=-1)
        x_graph = x_graph.view(batch_size, -1)

        # concatenate
        print(x_sample.device())
        print(x_graph.device())
        print(batch_size.device())
        x = torch.cat((x_sample, x_graph), dim=1).view(batch_size, 1, -1).to(device)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x).view(batch_size, 128)
        x = self.fc1(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)
        

    def get_edge_index(self, x):
        edge_index_1 = []
        edge_index_2 = []
        batch_size = x.shape[0]
        for b in range(batch_size):
            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    if pow(x[b][i][0]-x[b][j][0], 2) < MAX_DIFF:
                        edge_index_1.append(2000*b+i); edge_index_1.append(2000*b+j)
                        edge_index_2.append(2000*b+j); edge_index_2.append(2000*b+i)

        return torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)






        
