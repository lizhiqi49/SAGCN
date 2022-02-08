#%%
#conding:utf-8

# lzq
# match the samples with clinical information
# 2021-03-23

import numpy as np
import pandas as pd

sample_alignment_df = pd.read_csv('sample_alignment.csv',index_col=0)
sample_tsv_df = pd.read_csv('G:/SJTU/WeiLab/clinicalData/biospecimen/sample.tsv',sep='\t')
clinical_tsv_df = pd.read_csv('G:/SJTU/WeiLab/clinicalData/clinical/clinical.tsv',sep='\t')
exposure_tsv_df = pd.read_csv('G:/SJTU/WeiLab/clinicalData/clinical/exposure.tsv',sep='\t')


# %%
def dropBlankColumn(df):
    ncol = len(df.columns)
    drop_indexs = []
    for i in range(ncol):
        if (df.iloc[:,i].astype(np.str)=="'--").all():
            drop_indexs.append(int(i))
    df = df.drop(columns=df.columns[drop_indexs])
    return df

sample_dropped_df = dropBlankColumn(sample_tsv_df)
clinical_dropped_df = dropBlankColumn(clinical_tsv_df)
exposure_dropped_df = dropBlankColumn(exposure_tsv_df)

df_inner = pd.merge(sample_alignment_df,sample_dropped_df,how='inner',on=['sample_submitter_id','sample_type'])
df_inner = pd.merge(df_inner,clinical_dropped_df,how='inner',on=['case_id','case_submitter_id'])
df = pd.merge(df_inner,exposure_dropped_df,how='left',on=['case_id','case_submitter_id'])
df.to_csv('matched_info.csv')