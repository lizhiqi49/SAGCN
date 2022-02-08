#%%

# annotate the selected probes with informations from TCGA and GEO
# 2021.11.10
# ----------------------------------------------------------------

import numpy as np
import pandas as pd

tcga_annotation = pd.read_csv('../../data/aSample_TCGA.txt', sep='\t', index_col=0).iloc[:, 1:]
geo_annoatation = pd.read_csv('../../data/meth_annotation_geo.csv', index_col=0).iloc[:, 1:]

selected_probes = pd.read_csv('../../data/selected_probes_name_350.csv').values.reshape(-1)
probes_tcga_anno = tcga_annotation.loc[selected_probes, :]
probes_geo_anno = geo_annoatation.loc[selected_probes, :]

probes_anno = pd.concat([probes_tcga_anno, probes_geo_anno], axis=1)
probes_anno.to_csv('../../data/selected_probes_350_annotation.csv')
# %%
