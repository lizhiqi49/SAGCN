#coding:utf-8

# lzq
# 1. filter the probes whose beta value is 0 in all samples
# 2. filter the probes that located on sex chromosomes 
# 3. filter the probes not mapping uniquely to human reference genome 19 (n=3965)
# 4. filter the probes containing single-nucleotide polymorphisms (dbSNP132Common, n=7998)
# 5. filter the probes not included on the Illumina EPIC (850k) array (n=32260)
# 2021-09-24


from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
import os


# === get the indexes of lines that are all 0 in beta value matrix
dir = '/lustre/home/acct-clsdqw/clsdqw-jiangxue/TCGA/methylation/'

basefiledir = '/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/file_alignment.csv'
basefile = pd.read_csv(basefiledir,index_col=0)
basefile = basefile.sort_values(by='Site Code')     # sorted by "primary site"
basefile_values = basefile.values
filelist = basefile_values[:,0]    # the filenames

isnan = np.array([True]*485577)
for i in range(len(filelist)):
    print(i)
    subject_code = filelist[i]
    path = dir + subject_code
    df = pd.read_csv(path,sep='\t')
    beta_values = df['Beta_value']
    isnan = isnan & np.isnan(beta_values).values

nanlines = np.where(isnan)[0]
print("Num of NaN Lines:",len(nanlines))

pd.DataFrame(nanlines).to_csv("/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/NaNLineIndex.csv")


# === remove the 0-lines from beta value matrix
beta_value_matrix_df = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/beta_value_matrix.csv',index_col=0)
beta_value_matrix_df = beta_value_matrix_df.drop(beta_value_matrix_df.index[nanlines])
beta_value_matrix_df = beta_value_matrix_df.reset_index(drop=True)


# === remove the methylation sites that are on sex-chromosomes 
aSample_df = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/data/aSample.txt',sep='\t')        # read annotations from one sample file
print(df.shape)
filter_sites = beta_value_matrix_df.iloc[:,0].values
aSample_df = aSample_df[aSample_df['Composite Element REF'].isin(filter_sites)]
print(aSample_df.shape)
sites = aSample_df.iloc[:,0]
notsexChrSites = sites[(aSample_df['Chromosome'].isin(['chrX','chrY']))==False].values
print(len(notsexChrSites))
beta_value_matrix_df.set_index('composite element ref',inplace=True)
beta_value_matrix_df = beta_value_matrix_df.loc[notsexChrSites,:]
print(beta_value_matrix_df.shape)
#beta_value_matrix_df.to_csv('./data/filtered_meth_matrix.csv')

# === remove the probes not mapping uniquely to human reference genome 19 (n=3965)
# === remove the probes containing single-nucleotide polymorphisms (dbSNP132Common, n=7998)
# === remove the probes not included on the Illumina EPIC (850k) array (n=32260)
#beta_value_matrix_df = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/gnn_nosex/data/filter_zero_meth_matrix_nosex.csv')
probes = beta_value_matrix_df.iloc[:, 0]

not_mapping_uniquely_hg19 = pd.read_csv('./filter/amb_3965probes.vh20151030.txt', sep='\n').values
contain_dbSNP = pd.read_csv('./filter/snp_7998probes.vh20151030.txt', sep='\n').values
not_inclued_on_850k = pd.read_csv('./filter/epicV1B2_32260probes.vh20160325.txt', sep='\n').values
removing_probes = np.concatenate((not_mapping_uniquely_hg19, contain_dbSNP, not_inclued_on_850k))
removing_probes = np.unique(removing_probes)

beta_value_matrix_df.set_index('composite element ref', inplace=True)
maintained_probes = probes[probes.isin(removing_probes).values == False]
beta_value_matrix_df = beta_value_matrix_df.loc[maintained_probes, :]
print(beta_value_matrix_df.shape)
beta_value_matrix_df.to_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/new_edit/preprocessing/data/filtered_meth_matrix.csv')