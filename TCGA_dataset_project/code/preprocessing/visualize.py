#%%
# lzq
# 2021.05.21
# visualize the results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SAVE_PATH = 'G:/SJTU/WeiLab/May/figs/'
#%%
# number of simples of different class
plt.figure(figsize=(15,8))
sample_numbers = [416, 653, 786, 839, 304, 439, 666, 409, 491, 396, 506, 596]
classes = ['Bladder','Brain','Breast','Bronchus','Cervix','Corpus','Kidney','Liver','Prostate','Stomach','Thyroid','Normal']
colors = ['blue','orange','green','red','purple','saddlebrown','deeppink','gray','gold','aqua','black','cadetblue']
palette = sns.color_palette(colors)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14)
plt.xlabel('Sample classes', fontsize=16, labelpad=10)
plt.ylabel('Number of samples', fontsize=16, labelpad=10)
plt.title('Sample sizes of different classes', fontsize=20, pad=14)
plot_sample_num = sns.barplot(x=classes, y=sample_numbers, palette=palette)
fig_sample_num = plot_sample_num.get_figure()
fig_sample_num.savefig(SAVE_PATH+'plot_sample_numbers.png', dpi=600)




#%%
base_dir = "G:/SJTU/WeiLab/May/compare_diffTissue/metrics/"

sites_50_df = pd.read_csv(base_dir+'metrics_diffTissue_50sites.csv', index_col=0)
sites_100_df = pd.read_csv(base_dir+'metrics_diffTissue_100sites.csv', index_col=0)
sites_150_df = pd.read_csv(base_dir+'metrics_diffTissue_150sites.csv', index_col=0)
sites_200_df = pd.read_csv(base_dir+'metrics_diffTissue_200sites.csv', index_col=0)
sites_250_df = pd.read_csv(base_dir+'metrics_diffTissue_250sites.csv', index_col=0)

metrics = np.concatenate((sites_50_df.values[-2, :], sites_100_df.values[-2, :], sites_150_df.values[-2, :], 
                          sites_200_df.values[-2, :], sites_250_df.values[-2, :]), axis=0).reshape((5,5))
metrics_df = pd.DataFrame(metrics, columns=['Accuracy', 'Recall', 'Precision', 'F1-score', 'AUC'], 
                          index=['k=0.1', 'k=0.2', 'k=0.3', 'k=0.4', 'k=0.5'])
#%%
metrics_sns_df = pd.DataFrame({'score': metrics_df.values.ravel(),
                               'metric': ['Accuracy', 'Recall', 'Precision', 'F1-score', 'AUC']*5,
                               'k': np.array(['0.1', '0.2', '0.3', '0.4', '0.5']*5).reshape((5,5)).T.ravel()})
colors = ['blue', 'red', 'green', 'gold', 'auqa']
#palette = sns.color_palette(colors)
plt.figure(figsize=(10,8))
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14)
plt.xlabel('Value of k', fontsize=16, labelpad=10)
plt.ylabel('Score', fontsize=16, labelpad=10)
plt.title("Metrics' value of under different k", fontsize=20, pad=14)
lineplot_metrics_diffTissue = sns.lineplot(x="k", y="score", hue="metric", 
                                           style="metric", data=metrics_sns_df)
fig_metrics_diffTissue = lineplot_metrics_diffTissue.get_figure()
fig_metrics_diffTissue.savefig(SAVE_PATH + 'lineplot_metrics_diffTissue.png', dpi=600)
# %%
