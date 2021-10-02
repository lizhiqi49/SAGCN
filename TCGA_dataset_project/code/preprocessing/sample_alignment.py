import numpy as np
import pandas as pd
import os

def sample_alignment(file_name, bladder, brain, breast, bronchus_and_lung, cervix, corpus_uteri, kidney, liver, prostate_gland, stomach, thyroid_gland):

    r1, c1 = bladder.shape
    r2, c2 = brain.shape
    r3, c3 = breast.shape
    r4, c4 = bronchus_and_lung.shape
    r5, c5 = cervix.shape
    r6, c6 = corpus_uteri.shape
    r7, c7 = kidney.shape
    r8, c8 = liver.shape
    r9, c9 = prostate_gland.shape
    r10, c10 = stomach.shape
    r11, c11 = thyroid_gland.shape

    fileName = []
    fileid = []
    sampleid = []
    sampleType = []
    primarySite = []
    siteCode = []

    #counter_allsample = 0

    for j in range(len(file_name)):
        isFound = False

        if not isFound:
            for i in range(r1):
                if file_name[j] == bladder[i,1]:
                    isFound = True
                    fileName.append(bladder[i,1])
                    fileid.append(bladder[i,0])
                    sampleid.append(bladder[i,-2])
                    sampleType.append(bladder[i,-1])
                    primarySite.append("bladder")
                    siteCode.append(1)
                    print(j, "bladder")
                    break
        
        if not isFound:
            for i in range(r2):
                if file_name[j] == brain[i,1]:
                    isFound = True
                    fileName.append(brain[i,1])
                    fileid.append(brain[i,0])
                    sampleid.append(brain[i,-2])
                    sampleType.append(brain[i,-1])
                    primarySite.append("brain")
                    siteCode.append(2)
                    print(j, "brain")
                    break

        if not isFound:
            for i in range(r3):
                if file_name[j] == breast[i,1]:
                    isFound = True
                    fileName.append(breast[i,1])
                    fileid.append(breast[i,0])
                    sampleid.append(breast[i,-2])
                    sampleType.append(breast[i,-1])
                    primarySite.append("breast")
                    siteCode.append(3)
                    print(j, "breast")
                    break

        if not isFound:
            for i in range(r4):
                if file_name[j] == bronchus_and_lung[i,1]:
                    isFound = True
                    fileName.append(bronchus_and_lung[i,1])
                    fileid.append(bronchus_and_lung[i,0])
                    sampleid.append(bronchus_and_lung[i,-2])
                    sampleType.append(bronchus_and_lung[i,-1])
                    primarySite.append("bronchus and lung")
                    siteCode.append(4)
                    print(j, "bronchus and lung")
                    break

        if not isFound:
            for i in range(r5):
                if file_name[j] == cervix[i,1]:
                    isFound = True
                    fileName.append(cervix[i,1])
                    fileid.append(cervix[i,0])
                    sampleid.append(cervix[i,-2])
                    sampleType.append(cervix[i,-1])
                    primarySite.append("cervix uteri")
                    siteCode.append(5)
                    print(j, "cervix uteri")
                    break

        
        if not isFound:
            for i in range(r6):
                if file_name[j] == corpus_uteri[i,1]:
                    isFound = True
                    fileName.append(corpus_uteri[i,1])
                    fileid.append(corpus_uteri[i,0])
                    sampleid.append(corpus_uteri[i,-2])
                    sampleType.append(corpus_uteri[i,-1])
                    primarySite.append("corpus uteri")
                    siteCode.append(6)
                    print(j, "corpus uteri")
                    break

        if not isFound:
            for i in range(r7):
                if file_name[j] == kidney[i,1]:
                    isFound = True
                    fileName.append(kidney[i,1])
                    fileid.append(kidney[i,0])
                    sampleid.append(kidney[i,-2])
                    sampleType.append(kidney[i,-1])
                    primarySite.append("kidney")
                    siteCode.append(7)
                    print(j, "kidney")
                    break

        if not isFound:
            for i in range(r8):
                if file_name[j] == liver[i,1]:
                    isFound = True
                    fileName.append(liver[i,1])
                    fileid.append(liver[i,0])
                    sampleid.append(liver[i,-2])
                    sampleType.append(liver[i,-1])
                    primarySite.append("liver and intrahepatic bile ducts")
                    siteCode.append(8)
                    print(j, "liver and intrahepatic bile ducts")
                    break

        if not isFound:
            for i in range(r9):
                if file_name[j] == prostate_gland[i,1]:
                    isFound = True
                    fileName.append(prostate_gland[i,1])
                    fileid.append(prostate_gland[i,0])
                    sampleid.append(prostate_gland[i,-2])
                    sampleType.append(prostate_gland[i,-1])
                    primarySite.append("prostate gland")
                    siteCode.append(9)
                    print(j, "prostate gland")
                    break


        if not isFound:
            for i in range(r10):
                if file_name[j] == stomach[i,1]:
                    isFound = True
                    fileName.append(stomach[i,1])
                    fileid.append(stomach[i,0])
                    sampleid.append(stomach[i,-2])
                    sampleType.append(stomach[i,-1])
                    primarySite.append("stomach")
                    siteCode.append(10)
                    print(j, "stomach")
                    break

        if not isFound:
            for i in range(r11):
                if file_name[j] == thyroid_gland[i,1]:
                    isFound = True
                    fileName.append(thyroid_gland[i,1])
                    fileid.append(thyroid_gland[i,0])
                    sampleid.append(thyroid_gland[i,-2])
                    sampleType.append(thyroid_gland[i,-1])
                    primarySite.append("thyroid gland")
                    siteCode.append(11)
                    print(j, "thyroid gland")
                    break
    
    fileName = np.array(fileName).reshape((len(fileName),1))
    fileid = np.array(fileid).reshape((len(fileid),1))
    sampleid = np.array(sampleid).reshape((len(sampleid),1))
    sampleType = np.array(sampleType).reshape((len(sampleType),1))
    primarySite = np.array(primarySite).reshape((len(primarySite),1))
    siteCode = np.array(siteCode).reshape((len(siteCode),1))

    f_alignment = np.hstack((fileName,fileid))
    f_alignment = np.hstack((f_alignment,sampleid))
    f_alignment = np.hstack((f_alignment,sampleType))
    f_alignment = np.hstack((f_alignment,primarySite))
    f_alignment = np.hstack((f_alignment,siteCode))
    #f_alignment = np.vstack((colnames,f_alignment))

    return f_alignment

def main():
    # 获取所有甲基化数据的文件名
    dir = "/lustre/home/acct-clsdqw/clsdqw-jiangxue/TCGA/methylation/"
    meth_filename = os.listdir(dir)
    colnames = np.array(['File Name','File ID','Sample ID','Sample Type','Primary Site','Site Code'])

    # 读入11种primary site 的文件

    bladder = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/bladder.csv').values
    brain = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/brain.csv').values
    breast = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/breast.csv').values
    bronchus_and_lung = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/bronchus_and_lung.csv').values
    cervix = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/cervix_uteri.csv').values
    colon = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/colon.csv').values
    corpus_uteri = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/corpus_uteri.csv').values
    kidney = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/kidney.csv').values
    liver = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/liver_and_intrahepatic_bile_ducts.csv').values
    prostate_gland = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/prostate_gland.csv').values
    stomach = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/stomach.csv').values
    thyroid_gland = pd.read_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/preprocess_data/finish/thyroid_gland.csv').values

    f_alignment = sample_alignment(meth_filename, bladder, brain, breast, bronchus_and_lung, cervix, corpus_uteri, kidney, liver, prostate_gland, stomach, thyroid_gland)
    f_alignment_df = pd.DataFrame(data = f_alignment, columns=colnames)
    f_alignment_df = f_alignment_df.sort_values
    f_alignment_df.to_csv('/lustre/home/acct-clsdqw/clsdqw-jiangxue/lzq/dataPreprocess/sample_alignment.csv')

if __name__ == '__main__':
    main()