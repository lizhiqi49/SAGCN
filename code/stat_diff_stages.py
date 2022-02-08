#%%

# statistic the prediction results of tumor samples of different stages
# 2021.11.24

# ----------------------------------------------------------------


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

proba_file = '../prediction_results/clinical_sample_classification/result_svc_350probes.csv'
sample_label_file = '../data/sample_label_random.csv'
sample_info_file = '../data/merged_file_sorted.csv'

results = pd.read_csv(proba_file, index_col=0)
sample_label_random = pd.read_csv(sample_label_file, index_col=0)
sample_stage = pd.read_csv(sample_info_file, index_col=0).loc[:, ['File Name', 'sample_type', 'tumor_stage']]

# %%
sample_stage.columns = ['Sample', 'Type', 'Stage']
sample_label_type_stage = pd.merge(sample_label_random, sample_stage, how='left', on='Sample')
y = sample_label_random.values[:, -1]
sample_info_order_as_results = []
cv = KFold(n_splits=5, shuffle=False)
for fold, (train_ind, test_ind) in enumerate(cv.split(y)):
    sample_info_order_as_results += sample_label_type_stage.values[test_ind].tolist()
sample_info_order_as_results = np.array(sample_info_order_as_results)

results.index = sample_info_order_as_results[:, 0]

#%%

flag1 = (sample_label_type_stage['Type'] == 'Primary Tumor').values & (sample_label_type_stage['Stage'] == 'stage i').values
samples_stage1 = sample_label_type_stage[flag1]['Sample'].values
results_stage1 = results.loc[samples_stage1, :].values
flag2 = (sample_label_type_stage['Type'] == 'Primary Tumor').values & (sample_label_type_stage['Stage'] == 'stage ii').values
samples_stage2 = sample_label_type_stage[flag2]['Sample'].values
results_stage2 = results.loc[samples_stage2, :].values


# %%
from sklearn.metrics import *



def onehotCoding(arr, num_classes):
    return np.eye(num_classes)[arr]


# === compute the metrics of each class and their macro/micro average
def computeMetrics(results_of_one_stage, output_path):
    primary_sites = ['Bladder', 'Brain', 'Breast', 'Bronchus & Lung',
                     'Cervix', 'Corpus', 'Kidney', 'Liver', 'Prostate', 'Stomach', 'Thyroid']
    primary_sites = np.array(primary_sites)
    results = []
    types_ = []

    y_real = results_of_one_stage[:, -1].astype(int)
    y_scores = results_of_one_stage[:, :12]
    y_pred = np.argmax(y_scores, axis=1)
    y_preds = onehotCoding(y_pred, 12)
    y_reals = onehotCoding(y_real, 12)
    n_sample, n_class = y_reals.shape
    n_class = 11

    accs = []
    recalls = []
    precisions = []
    f1s = []
    brier_scores = []
    log_losses = []
    aucs = []
    aupr = []
    recall_dict = dict()
    precision_dict = dict()

    for i in range(n_class):
        if (y_reals[:, i] == 0).all():

            continue
        types_.append(i)

        accs.append(accuracy_score(y_reals[:, i], y_preds[:, i]))
        recalls.append(recall_score(y_reals[:, i], y_preds[:, i]))
        precisions.append(precision_score(y_reals[:, i], y_preds[:, i]))
        f1s.append(f1_score(y_reals[:, i], y_preds[:, i]))
        brier_scores.append(brier_score_loss(    y_reals[:, i], y_scores[:, i]))
        log_losses.append(log_loss(y_reals[:, i], y_scores[:, i]))
        aucs.append(roc_auc_score(y_reals[:, i], y_scores[:, i]))
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_reals[:, i], y_scores[:, i])
        pr_auc = auc(recall_dict[i], precision_dict[i])
        aupr.append(pr_auc)

    accs = np.array(accs, dtype=np.float)
    mes = np.ones_like(accs)-accs
        

    metric_df = pd.DataFrame({'ME': mes, 'Brier Score': brier_scores, 'Log Loss': log_losses,
                            'Recall': recalls, 'Precision': precisions, 'F1-score': f1s,
                            'AUC': aucs, 'AUPR': aupr}, index=primary_sites[types_])
    

    metric_df.to_csv(output_path)

computeMetrics(results_stage1, './perf_eval_stage1.csv')
computeMetrics(results_stage2, './perf_eval_stage2.csv')

# %%
