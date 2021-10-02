#%%
# lzq
# the data process from filtered methylation matrix

from subfunctions import *

filtered_meth_matrix_path = './data/filtered_meth_matrix.csv'
meth_matrix_maxstd_path = './data/meth_matrix_maxstd_2k_sorted.csv'
pca_result_path = './data/pca_top100.csv'
tsne_fig_path = './data/tsne.png'

#subfuncs_maxstd_2k(filtered_meth_matrix_path, meth_matrix_maxstd_path)
#subfuncs_pca(meth_matrix_maxstd_path, pca_result_path)
subfuncs_tsne(pca_result_path, tsne_fig_path)
# %%
