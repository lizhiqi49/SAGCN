#coding:utf-8

# lzq
# select the top 2k methylation sites with max standard deviation



import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

def subfuncs_maxstd_2k(filtered_meth_matrix_file_path, save_meth_matrix_maxstd_path, top_std=2000):
    r"""
    # select the top 2k methylation sites with max standard deviation
    """
    #matdir = '/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/beta_value_matrix_filter_zero.csv'
    beta_matrix_df = pd.read_csv(filtered_meth_matrix_file_path)    # 读取甲基化矩阵
    meth_site_num = len(beta_matrix_df)         # 甲基化位点总数
    beta_values = beta_matrix_df.values[:,1:].astype(np.float)

    stds = []
    for row in beta_values:
        std = np.std(row)
        stds.append(std)

    beta_matrix_df['standard deviation'] = stds     # 甲基化矩阵添加新列std
    beta_matrix_df = beta_matrix_df.sort_values(by='standard deviation',ascending=False)    # 按std降序排序
    beta_matrix_df_max20k = beta_matrix_df.head(top_std).reset_index(drop=True).iloc[:,:-1]       # 提取std最大的前20k
    beta_matrix_df_max20k.to_csv(save_meth_matrix_maxstd_path)

def subfuncs_pca(meth_matrix_maxstd_file_path, pca_result_file_path, top_pc=100):
    meth_matrix_max2k_df = pd.read_csv(meth_matrix_maxstd_file_path, index_col=0)
    meth_matrix_max2k_T = meth_matrix_max2k_df.values[:,1:].T
    print(meth_matrix_max2k_T.shape)

    pca = PCA(n_components=top_pc)
    pca.fit(meth_matrix_max2k_T)
    print("explained variance ratio:",pca.explained_variance_ratio_)
    pca_vectors_top100 = pca.transform(meth_matrix_max2k_T).T
    col_name = meth_matrix_max2k_df.columns[1:]
    pca_top100_df = pd.DataFrame(pca_vectors_top100,columns=col_name)
    pca_top100_df.to_csv(pca_result_file_path)

def subfuncs_tsne(pca_result_file_path, tsne_fig_save_path):
    pca_mat = pd.read_csv(pca_result_file_path,index_col=0)
    X = pca_mat.values.T    # shape=(6900*100)
    #labels = [0]*438 + [1]*686 + [2]*886 + [3]*914 + [4]*309 + [5]*343 + [6]*474 + [7]*872 + [8]*468 + [9]*542 + [10]*398 + [11]*570
    labels = ['Bladder']*416 + ['Brain']*653 + ['Breast']*786 + ['Bronchus']*839 + ['Cervix']*304 + \
        ['Corpus uteri']*439 + ['Kidney']*666 + ['Liver']*409 + ['Prostate']*491 + ['Stomach']*396 + ['Thyroid']*506 + ['Normal']*596
    labels = np.array(labels)
    tsne = TSNE()
    #%%
    X_embedded = tsne.fit_transform(X)

    colors = ['blue','orange','green','red','purple','saddlebrown','deeppink','gray','gold','aqua','black','cadetblue']
    sns.set_context('paper')
    sns.set(rc={'figure.figsize':(15,10)})
    palette = sns.color_palette(colors)
    sns.set_style('white')

    fig = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=labels, legend='full', palette=palette)
    fig.legend(loc='upper right',ncol=2,borderaxespad=0.3,fontsize=14,markerscale=1.5)
    plt.xlabel('tSNE1',fontsize=16,fontweight='bold')
    plt.ylabel('tSNE2',fontsize=16,fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    plt.rcParams['figure.dpi'] = 600 #分辨率

    scatter_fig = fig.get_figure()
    scatter_fig.savefig(tsne_fig_save_path, dpi = 600)





    