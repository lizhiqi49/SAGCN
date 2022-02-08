#%%

# select the k_3 * 500 probes that show up most frequently
# in all samples' DEMs output from SAGCN
# --------------------------------------------------------

import numpy as np
import pandas as pd
from collections import Counter

"""
num_probes = [50, 100, 150, 200, 300, 350]
for num_probe in num_probes:
    meth_matrix_maxstd_2k = pd.read_csv('../../../compared_methods/meth_matrix_maxstd_2k_sorted.csv', index_col=0)
    perms = pd.read_csv('../../prediction_results/sagcn_results/perms_{}sites.csv'.format(str(num_probe)), index_col=0)
    counter = Counter(perms.values.flatten())
    keys_list = np.array(list(dict(counter).keys()))
    values_list = np.array(list(dict(counter).values()))

    order = values_list.argsort()[::-1]
    probes_selected = keys_list[order][:num_probe]

    meth_matrix_probes_150 = meth_matrix_maxstd_2k.iloc[probes_selected, :]
    meth_matrix_probes_150.to_csv('../../data/meth_matrix_probes_{}.csv'.format(str(num_probe)))

"""






# %%

## draw heatmap
import matplotlib.pyplot as plt
import seaborn as sns



meth_matrix_probes_150 = pd.read_csv('../../data/meth_matrix_probes_350.csv', index_col=0).set_index('composite element ref', drop=True)
plt.style.use('ggplot')
plt.figure(figsize=(24, 16))
fig = sns.heatmap(meth_matrix_probes_150, cmap='rainbow', xticklabels=False)
fig = fig.get_figure()
fig.savefig('../../data/heatmap_probes_350_rainbow.png', dpi=300)
#fig = sns.clustermap(meth_matrix_probes_150, row_cluster=False, xticklabels=False, cmap='PuBuGn', figsize=(25, 12))



# %%
