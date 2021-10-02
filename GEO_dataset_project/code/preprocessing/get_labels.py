#%%

# lzq
# get the labels of samples and transform them into integers
# ----------------------------------------------------------------

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


sample_info = pd.read_csv('./Supplementary_table_2.csv')
sample_class_map = sample_info.loc[:, ['Sentrix ID (.idat)', 'Reference Group abbreviation']]
# %%
class_dict = sample_class_map['Reference Group abbreviation'].unique().tolist()
sample_class_map['label'] = sample_class_map['Reference Group abbreviation'].apply(lambda x : class_dict.index(x))
# %%
sample_class_map.to_csv('./sample_label_sorted.csv')
# %%
counter = Counter(sample_class_map.values[:, -1])
keys_list = np.array(list(dict(counter).keys()))
values_list = np.array(list(dict(counter).values()))

order = values_list.argsort()[::-1]
labels_sorted = keys_list[order]

classes_sorted = sample_class_map.iloc[:, 1].unique()[labels_sorted]
sample_size = values_list[order]
df = pd.DataFrame({'class': classes_sorted, 
                   'label': labels_sorted,
                   'sample_size': sample_size})

#%%
sns.set_style("white")


plt.figure(figsize=(10, 6))
plt.rcParams['figure.dpi'] = 600
fig = sns.barplot(x=list(range(91)), y='sample_size', data=df, )
plt.xticks([])
plt.ylabel("Sample Size")
plt.yticks(fontsize=(10))
fig = fig.get_figure()
fig.savefig('./sample_distribution.png', dpi=600)

# %%
meth_mat = pd.read_csv('./Mset_maxstd_2k.csv', index_col=0)
columns = np.array(meth_mat.columns, dtype=np.str)
#%%
for i, element in enumerate(columns):
    columns[i] = columns[i][11:]
# %%
meth_mat = pd.DataFrame(meth_mat.values, columns = columns)
meth_mat = meth_mat.loc[:, sample_class_map.values[:, 0]]
meth_mat.to_csv('./Mset_maxstd_2k_sorted.csv')
# %%
