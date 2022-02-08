
# lzq
# perform statistic for clinical data on matched_info.csv


# %%
from collections import Counter
import numpy as np
import pandas as pd
#d = Counter(df['Primary Site'])
# %%
df = pd.read_csv('matched_info.csv',index_col=0)
primary_sites = ['bladder','breast','breast','bronchus and lung','cervix uteri','corpus uteri','kidney',\
                 'liver and intrahepatic bile ducts','prostate gland','stomach','thyroid gland']

#%%
def getInfo():
    info = np.array([])
    for primary_site in primary_sites:
        data = df[df['Primary Site']==primary_site]
        info = np.hstack((info,getSampleInfo(data))
        normal_df = df[df['sample_type']=='Solid Tissue Normal']
        info = np.hstack((info,getSampleInfo(normal_df))
        case_df = df[df['sample_type']!='Solid Tissue Normal']
        np.hstack((info,getSampleInfo(case_df))
    info_df = pd.DataFrame(info)
    info_df.to_csv('clinical_info_statistic.csv')
        


def getSampleInfo(df):
    mean_age, std_age = getAge(df)
    ages = catMeanAndStd([mean_age], [std_age])
    n_female, ratio_female = getFemale(df)
    female = catNumAndRatio([n_female], [ratio_female])
    n_race, ratio_race = getRace(df)
    race = catNumAndRatio(n_race, ratio_race)
    n_age65, ratio_age65 = getAgeGreater65(df)
    age65 = catNumAndRatio([n_age65], [ratio_age65])
    n_stage, ratio_stage = getStage(df)
    stage = catNumAndRatio(n_stage, ratio_stage)
    info_arr = np.concatenate((ages,female,race,age65,stage))
    return info_arr


def getAge(df):
    rows_not_null = df[df['age_at_index']!="'--"]['age_at_index'].values
    mean = np.mean(rows_not_null.astype(np.float))
    std = np.std(rows_not_null.astype(np.float))
    return mean, std

def getFemale(df):
    n_female = Counter(df['gender'])['female']
    ratio_female = n_female/len(df)
    return n_female, ratio_female

def getRace(df):
    n_white = Counter(df['race'])['white']
    ratio_white = n_white / len(df)
    n_african = Counter(df['race'])['black or african american']
    ratio_african = n_african / len(df)
    n_asian = Counter(df['race'])['asian']
    ratio_asian = n_asian / len(df)
    n_other = Counter(df['race'])['american indian or alaska native'] + \
        Counter(df['race'])['native hawaiian or other pacific islander'] + \
            Counter(df['race'])['not reported']
    ratio_other = n_other / len(df)
    n = [n_white, n_african, n_asian, n_other]
    ratio = [ratio_white, ratio_african, ratio_asian, ratio_other]
    return n, ratio


def getAgeGreater65(df):
    rows_not_null = df[df['age_at_index']!="'--"]
    n_age65 = len(rows_not_null[rows_not_null['age_at_index'].astype(np.int)>65])
    ratio = n_age65/len(df[df['age_at_index']!="'--"])
    return n_age65, ratio

def getStage(df):
    dict = Counter(df['ajcc_pathologic_stage'])
    n1 = dict['Stage I'] + dict['Stage IA'] + dict['Stage IB']
    ratio1 = n1 / len(df)
    n2 = dict['Stage II'] + dict['Stage IIA'] + dict['Stage IIB']
    ratio2 = n2 / len(df)
    n3 = dict['Stage III'] + dict['Stage IIIA'] + dict['Stage IIIB'] + dict['Stage IIIC']
    ratio3 = n3 / len(df)
    n4 = dict['Stage IV'] + dict['Stage IVA'] + dict['Stage IVB'] + dict['Stage IVC']
    ratio4 = n4 / len(df)
    n_other = len(df) - n1 - n2 - n3 - n4
    ratio_other = n_other / len(df)
    ns = [n1, n2, n3, n4, n_other]
    ratios = [ratio1, ratio2, ratio3, ratio4, ratio_other]
    return ns, ratios

def catNumAndRatio(n,ratio):
    length = len(n)
    cat = []
    for i in range(length):
        s = str(n[i]) + '(' + str("{:.1f}".format(ratio[i]*100)) + '%)'
        cat.append(s)
    return np.array(cat)

def catMeanAndStd(n,ratio):
    length = len(n)
    cat = []
    for i in range(length):
        s = str("{:.2f}".format(n[i])) + '+/-' + str("{:.2f}".format(ratio[i]))
        cat.append(s)
    return np.array(cat)

getInfo()
# %%
