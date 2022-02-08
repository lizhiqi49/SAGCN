#%%

# lzq
# 2021.05.10
# ===================================================
#   run this script to create graph dataset
#   the dataset is stored in the path "[root]/processed/..."
# ===================================================

import random
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset


EDGE_THRESHOLD = 0.05
NUM_SAMPLING = 200      # the sampling times when creating a graph
NUM_CLASSES = 12
REPEAT = 50     # the numbers of created graphs for each class
ROOT_PATH = './val'
BASE_DIR = 'G:/SJTU/cancerMethy/new_edit/preprocessing/data/'   # the dir of methylation matrix file

class MyInMemoryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        ## root: the path to save dataset

        super(MyInMemoryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['meth_matrix_maxstd_2k_sorted.csv']

    @property
    def processed_file_names(self):
        return ['dataset_2k.pt']

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):
        # Read data into huge `Data` list.
        basedir = BASE_DIR
        #samplename_sorted = pd.read_csv(basedir+'merged_file_sorted.csv',index_col=0).reset_index(drop=True).values[:,0]
        meth_mat_maxstd_2k_df = pd.read_csv(
            basedir + self.raw_file_names[0], index_col=0)
        #meth_mat_maxstd_20k_df = pd.read_csv(basedir+'meth_matrix_maxstd_500.csv',index_col=0)
        meth_sites = meth_mat_maxstd_2k_df.values[:, 0]
        beta_values = meth_mat_maxstd_2k_df.values[:, 1:].T

        data_list = []
        sampling_datas = sampling(beta_values)
        n_type, n_repeat, n_site, n_sample = sampling_datas.shape
        print("sampling data shape:",sampling_datas.shape)

        powers = np.array([5, 3, 8, 6, 3, 4, 7, 5, 2, 5, 3, 10])

        for i in range(n_type):
            #power, edge_threshold = powerAndThreshold(int(i))
            power = powers[i]
            for j in range(n_repeat):
                g = wgcnaNet(sampling_datas[i, j],power,int(i))
                data_list.append(g)

        print("num of graphs:", len(data_list))

        if self.pre_filter is not None:
            data_list = [
                data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slice = self.collate(data_list)
        torch.save((data, slice), osp.join(self.processed_dir, 'dataset_2k.pt'))


def sampling(beta_values):
    n_sample = NUM_SAMPLING
    n_repeat = REPEAT

    def sampleFromData(mat):
        return sampleFromData_(mat, n_sample, n_repeat)

    # shape: (n_repeat,20k,n_sample)
    bladder = sampleFromData(beta_values[:416])
    brain = sampleFromData(beta_values[416:1069])
    breast = sampleFromData(beta_values[1069:1855])
    bronchus = sampleFromData(beta_values[1855:2694])
    cervix = sampleFromData(beta_values[2694:2998])
    #colon = sampleFromData(beta_values[3233:3576])
    corpus = sampleFromData(beta_values[2998:3437])
    kidney = sampleFromData(beta_values[3437:4103])
    liver = sampleFromData(beta_values[4103:4512])
    prostate = sampleFromData(beta_values[4512:5003])
    stomach = sampleFromData(beta_values[5003:5399])
    thyroid = sampleFromData(beta_values[5399:5905])
    normal = sampleFromData(beta_values[5905:])

    datas = np.array([bladder, brain, breast, bronchus,cervix,corpus,kidney,liver,prostate,stomach,thyroid,normal])     # shape: (12,n_repeat,20k,n_sample)
    return datas


def sampleFromData_(mat, n_sample, n_repeat):
    """
    params:
        beta_values: the mathylation matrix of one class, an element: (index_sample, index_probe)
        n_sampling: the sampling times with replace when creating a graph
        n_repeat: the number of graphs to be created
    """
    data = []
    for i in range(n_repeat):
        choice_index = np.random.choice(len(mat), n_sample)
        data.append(mat[choice_index].T.tolist())
    return np.array(data)


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
    g = Data(x=X, edge_index=edge_index, y=y)
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


def main():
    dataset = MyInMemoryDataset(root=ROOT_PATH)
    

#%%
if __name__ == '__main__':
    main()

# %%
