#!/usr/bin/python
#coding:utf-8
"""
作者：jx
版本：1
文件名：read_beta_value.py
功能：将7544个样本的包括14种tumor的beta_value读入到一个文件中
日期：2020-08-22
"""
import pandas as pd
import numpy as np
import os


# 读取文件名及beta value值
def read_data(file_path):
    """
    :param file_path: 文件夹路径
    :return:
    """
    print('file path', file_path)
    subject_name = os.path.basename(file_path)
    data_df = pd.read_csv(file_path, sep = '\t')
    data_df = data_df.fillna(0)
    data = data_df.values

    methylation_name = data[:, 0]
    methylation_name = np.array(methylation_name).reshape(len(methylation_name), 1)
    beta_value = data[:, 1]
    beta_value = np.array(beta_value).reshape(len(beta_value), 1).astype(np.float)

    print(beta_value.shape)

    return subject_name, methylation_name, beta_value

def main():

    # ===== Step 1. 读入数据
    dir = '/lustre/home/acct-clsdqw/clsdqw-jiangxue/TCGA/methylation/'
    
    basefiledir = '/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/sample_alignment.csv'
    basefile = pd.read_csv(basefiledir,index_col=0)
    basefile = basefile.sort_values(by='Site Code')     # 按primary site排序
    basefile_values = basefile.values
    filelist = basefile_values[:,0]    #按primary site排序后所有文件的文件名
    #fileids = np.insert(basefile_values[:,1],0,'reference_methylation_name')      #所有文件的文件id
    
    print("file number:", len(filelist))

    file_name = []
    beta_value_matrix = []
    tmp = dir + filelist[0]
    reference_subject_name, reference_methylation_name, reference_beta_value = read_data(tmp)

    file_name.append('composite element ref')
    file_name.append(reference_subject_name)
    beta_value_matrix = reference_beta_value
    r = len(reference_methylation_name)
    print('reference data length:', r)

    for i in range(len(filelist)):
        print(i)
        subject_code = filelist[i]

        path = dir + subject_code

        subject_name, methylation_name, beta_value = read_data(path)

        beta_value_matrix = np.hstack((beta_value_matrix, beta_value))
        file_name.append(subject_name)


    # # ===== Step 2. 为数据矩阵增加行名和列名
    beta_value_matrix = np.hstack((reference_methylation_name, beta_value_matrix))
    #beta_value_matrix = np.vstack((file_name, beta_value_matrix))
    
    #file_name = np.array(file_name).reshape(len(file_name), 1)

    print('The final data shape is:', beta_value_matrix.shape)

    # ===== Step 3. 保存数据
    beta_value_matrix_df = pd.DataFrame(data = beta_value_matrix,columns=file_name)
    beta_value_matrix_df.to_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/beta_value_matrix.csv')

    #file_name_df = pd.DataFrame(data = file_name)
    #file_name_df.to_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/file_name.csv')

if __name__ == '__main__':
    main()



























