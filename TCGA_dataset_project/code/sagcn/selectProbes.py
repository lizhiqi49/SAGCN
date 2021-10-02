#%%

# select the k_3 * 500 probes that show up most frequently
# in all samples' DEMs output from SAGCN
# --------------------------------------------------------

import numpy as np
import pandas as pd
from collections import Counter


meth_matrix_maxstd_2k = pd.read_csv('../preprocessing/data/meth_matrix_maxstd_2k_sorted.csv', index_col=0)
perms = pd.read_csv('./results/perms_150sites.csv', index_col=0)
counter = Counter(perms.values.flatten())
keys_list = np.array(list(dict(counter).keys()))
values_list = np.array(list(dict(counter).values()))

order = values_list.argsort()[::-1]
probes_selected = keys_list[order][:150]

meth_matrix_probes_150 = meth_matrix_maxstd_2k.iloc[probes_selected, :]
meth_matrix_probes_150.to_csv('./results/meth_matrix_probes_150.csv')
# %%
