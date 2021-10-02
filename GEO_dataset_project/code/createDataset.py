#%%

# lzq
# 2021.09.27
# ===================================================
#   run this script to create graph dataset for external validation dataset
#   the dataset is stored in the path "[root]/processed/..."
# ===================================================

import random
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset

EDGE_POWER = 2
EDGE_THRESHOLD = 0.5
MINI_DISTURB_RATIO = 0.05       # the biggest ratio of mini-disturbance when sampling on one class with small size samples
NUM_SAMPLING = 40       # the sampling times when creating a graph
NUM_CLASSES = 91
REPEAT = 25     # the numbers of created graphs for each class
ROOT_PATH = './dataset'
BASE_DIR = './data/'   # the dir of methylation matrix file

class MyInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        ## root: the path to save dataset

        super(MyInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Mset_maxstd_2k_sorted.csv']

    @property
    def processed_file_names(self):
        return ['dataset_Mset_2k.pt']

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        basedir = BASE_DIR
        #samplename_sorted = pd.read_csv(basedir+'merged_file_sorted.csv',index_col=0).reset_index(drop=True).values[:,0]
        meth_mat_maxstd_2k_df = pd.read_csv(
            basedir + self.raw_file_names[0], index_col=0)
        sample_label_df = pd.read_csv(basedir + 'sample_label_sorted.csv', index_col=0)
        #meth_mat_maxstd_20k_df = pd.read_csv(basedir+'meth_matrix_maxstd_500.csv',index_col=0)
        #
        beta_values = meth_mat_maxstd_2k_df.values.T

        data_list = []
        sampling_datas = sampling(meth_mat_maxstd_2k_df, sample_label_df)
        n_type, n_repeat, n_site, n_sample = sampling_datas.shape
        print("sampling data shape:",sampling_datas.shape)


        for i in range(n_type):
            for j in range(n_repeat):
                g = wgcnaNet(sampling_datas[i, j], EDGE_POWER, int(i))
                data_list.append(g)

        print("num of graphs:", len(data_list))

        if self.pre_filter is not None:
            data_list = [
                data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slice = self.collate(data_list)
        torch.save((data, slice), osp.join(self.processed_dir, 'dataset_Mset_2k.pt'))


def sampling(meth_matrix_df, sample_label_df):
    n_classes = NUM_CLASSES
    n_sample = NUM_SAMPLING
    n_repeat = REPEAT

    def sampleFromData(mat):
        return sampleFromData_(mat, n_sample, n_repeat)

    datas = []
    for i in range(n_classes):
        samples = sample_label_df[sample_label_df['label']==0].values[:, 0]
        beta_values = meth_matrix_df.loc[:, samples].values.T
        data_sampling = sampleFromData(beta_values)     # shape:(n_repeat, 2k, n_sampling)
        datas.append(data_sampling)

    datas = np.array(datas)     # shape: (91, n_repeat, 2k, n_sampling)

    return datas


def sampleFromData_(beta_values, n_sampling, n_repeat):
    """
    params:
        beta_values: the mathylation matrix of one class, an element: (index_sample, index_probe)
        n_sampling: the sampling times with replace when creating a graph
        n_repeat: the number of graphs to be created
    """
    
    n_samples = len(beta_values)
    datas = []
    if n_samples > 80:
        for i in range(n_repeat):
            choice_index = np.random.choice(len(beta_values), n_sampling)
            datas.append(beta_values[choice_index].T.tolist())
    else:
        for i in range(n_repeat):
            data = []
            for j in range(n_sampling):
                choice = np.random.randint(0, n_samples)
                vec = miniDisturb(beta_values[choice], MINI_DISTURB_RATIO).tolist()
                data.append(vec)
            data = np.array(data)
            datas.append(data.T.tolist()) 

    return datas


def wgcnaNet(X, power, y):
    """
    params:
        X: the methylation matrix with which to create a graph, shape:(2000, n_sampling)
        power: soft-power
        edge_threshold
        y
    """
    edge_threshold = EDGE_THRESHOLD
    # Correlation coefficient matrix between nodes
    cor_mat = np.corrcoef(X)    # shape:(n_nodes,n_nodes)
    # adjacency matrix
    A = adjMatrix(cor_mat, power, edge_threshold)

    edge_index = edgeIndex(A)
    #edge_attr = edgeAttr(edge_index, cor_mat, power)
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor([y], dtype=torch.long)
    g = Data(x=X, edge_index=edge_index,  y=y)
    print(g)
    return g

def adjMatrix(cor_mat, power, threshold):
    A = cor_mat
    r, c = A.shape
    edge_attr = []
    for i in range(r):
        for j in range(c):
            if pow(A[i][j], power) > threshold:
                A[i][j] = 1
            else:
                A[i][j] = 0
    return A

def getEdgeNum(mat,power,threshold):
    cor_mat = np.corrcoef(mat)
    A = adjMatrix(cor_mat,power,threshold)
    edge_index = edgeIndex(A)
    return len(edge_index[0])


def edgeIndex(A):
    index1 = []
    index2 = []
    n = len(A)
    for i in list(range(n)):
        for j in list(range(n)):
            if i != j and A[i][j] == 1:
                index1.append(i)
                index2.append(j)
    index = torch.tensor([index1, index2], dtype=torch.long)
    return index


def edgeAttr(edge_index, cor_mat, power):
    edge_attr = []
    index1 = edge_index[0]
    index2 = edge_index[1]
    n = len(index1)
    for i in range(n):
        ind1 = index1[i]
        ind2 = index2[i]
        edge_attr.append(pow(cor_mat[ind1,ind2],power))

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_attr


def degMatrix(A):
    D = np.zeros(A.shape, dtype=np.int)
    n = len(A)
    for i in range(n):
        D[i, i] = np.sum(A[i])
    return D

def miniDisturb(v, ratio):
    length = len(v)
    miniratio = ratio * (np.random.rand(length) - 1) * 2
    mini_disturb = miniratio * v
    return v + mini_disturb


def main():
    dataset = MyInMemoryDataset(root=ROOT_PATH)
    


if __name__ == '__main__':
    main()

# %%
#dataset = MyInMemoryDataset(root=ROOT_PATH)
# %%
