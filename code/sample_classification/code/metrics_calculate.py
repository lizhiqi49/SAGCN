#%%
# lzq
# 2021.04.26
# compute the metrics

# accuracy
# macro precision
# macro recall
# macro F1
# AUC
# AUPR
# 
# ME (1-accuracy)
# brier score

import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def onehotCoding(arr, num_classes):
    return np.eye(num_classes)[arr]

def sum2one(arr):
    return arr / (np.sum(arr,axis=-1)).reshape((-1,1))
#%%
# === compute the metrics of each class and their macro/micro average
def computeMetrics(y_real_label, y_probas, output_path):
    cv = KFold(n_splits=5)
    results = []
    for train, test in cv.split(y_real_label):
        y_real = y_real_label[test]
        y_scores = y_probas[test]
        y_pred = np.argmax(y_scores,axis=1)
        y_preds = onehotCoding(y_pred,12)
        y_reals = onehotCoding(y_real,12)
        n_sample, n_class = y_reals.shape

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
            accs.append(accuracy_score(y_reals[:,i], y_preds[:,i]))
            recalls.append(recall_score(y_reals[:,i], y_preds[:,i]))
            precisions.append(precision_score(y_reals[:,i], y_preds[:,i]))
            f1s.append(f1_score(y_reals[:,i], y_preds[:,i]))
            brier_scores.append(brier_score_loss(y_reals[:, i], y_scores[:,i]))
            log_losses.append(log_loss(y_reals[:, i], y_scores[:,i]))
            aucs.append(roc_auc_score(y_reals[:,i], y_scores[:,i]))
            precision_dict[i], recall_dict[i], _ = precision_recall_curve(y_reals[:,i], y_scores[:,i])
            pr_auc = auc(recall_dict[i], precision_dict[i])
            aupr.append(pr_auc)

        accs.append(accuracy_score(y_real, y_pred))
        recalls.append(recall_score(y_real, y_pred, average='macro'))
        precisions.append(precision_score(y_real, y_pred, average='macro'))
        f1s.append(f1_score(y_real, y_pred, average='macro'))
        aucs.append(roc_auc_score(y_real, y_scores, average='macro', multi_class="ovo"))
        
        # First aggregate all false positive rates
        all_recall = np.unique(np.concatenate([recall_dict[i] for i in range(n_class)]))
        # Then interpolate all ROC curves at this points
        mean_precision = np.zeros_like(all_recall)
        for i in range(n_class):
            mean_precision += np.interp(all_recall, recall_dict[i], precision_dict[i])
        # Finally average it and compute AUC
        mean_precision /= n_class
        recall_dict["macro"] = all_recall
        precision_dict["macro"] = mean_precision
        aupr.append(auc(recall_dict["macro"], precision_dict["macro"]))


        accs.append(accuracy_score(y_real, y_pred))
        recalls.append(recall_score(y_real, y_pred, average='micro'))
        precisions.append(precision_score(y_real, y_pred, average='micro'))
        f1s.append(f1_score(y_real, y_pred, average='micro'))
        fpr_micro, tpr_micro, _ = roc_curve(y_reals.ravel(), y_scores.ravel())
        aucs.append(auc(fpr_micro, tpr_micro))
        precision, recall, _ = precision_recall_curve(y_reals.ravel(), y_scores.ravel())
        aupr.append(auc(recall, precision))

        brier_score_macro = np.mean(brier_scores)
        brier_scores.append(brier_score_macro)
        brier_score_micro = brier_score_macro
        brier_scores.append(brier_score_micro)

        log_loss_macro = np.mean(log_losses)
        log_losses.append(log_loss_macro)
        log_loss_micro = log_loss_macro
        log_losses.append(log_loss_micro)

        accs = np.array(accs, dtype=np.float)
        mes = np.ones_like(accs)-accs
        primary_sites = ['Bladder','Brain','Breast','Bronchus & Lung','Cervix','Corpus','Kidney','Liver','Prostate','Stomach','Thyroid','Normal']
        metric_df = pd.DataFrame({'Accuracy':accs, 'Recall':recalls, 
                            'Precision':precisions, 'F1-score':f1s, 
                            'Brier Score':brier_scores, 'Log Loss':log_losses, 'ME': mes,
                            'AUC':aucs, 'AUPR':aupr},
                                index = primary_sites + ['macro', 'micro'])
        results.append(metric_df.values.tolist())
    results = np.array(results)
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)
    means_df = pd.DataFrame(means, index=metric_df.index, columns=metric_df.columns+'_avg')
    stds_df = pd.DataFrame(stds, index=metric_df.index, columns=metric_df.columns+'_std')
    result_df = pd.concat([means_df, stds_df], axis=1)

    result_df.to_csv(output_path)

# %%
# === draw ROC
def drawROC(y_pred, y_real, n_classes, title, save_path):

    y = onehotCoding(y_real, n_classes)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    primary_sites = ['Bladder','Brain','Breast','Bronchus & Lung','Cervix','Corpus','Kidney','Liver','Prostate','Stomach','Thyroid','Normal']
    colors = ['blue','orange','green','red','purple','saddlebrown','deeppink','gray','gold','aqua','black','cadetblue']
    plt.figure(figsize=(10,10))


    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (AUC = {0:0.4f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (AUC = {0:0.4f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    lw = 2
    for i in list(range(n_classes)):
        plt.plot(fpr[i], tpr[i], color=colors[i],
                lw=lw, label=primary_sites[i] + '(AUC = {:.4f})'.format(roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.savefig(save_path ,dpi=600)

# === draw PR Curve
def drawPR(y_pred, y_real, n_classes, title, save_path):
    plt.figure(figsize=(10,10))
    colors = ['blue','orange','green','red','purple','saddlebrown','deeppink','gray','gold','aqua','black','cadetblue']
    primary_sites = ['Bladder','Brain','Breast','Bronchus & Lung','Cervix','Corpus','Kidney','Liver','Prostate','Stomach','Thyroid','Normal']

    y_real = onehotCoding(y_real, n_classes)
    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i in range(12):
        precision[i], recall[i], _ = precision_recall_curve(y_real[:,i], y_pred[:,i])
        pr_auc[i] = auc(recall[i], precision[i])

        plt.step(recall[i], precision[i], color=colors[i], lw=2, alpha=0.5,
                label=primary_sites[i] + '(AUC = {:.4f})'.format(pr_auc[i]))
        #plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    precision["micro"], recall["micro"], _ = precision_recall_curve(y_real.ravel(), y_pred.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    # First aggregate all false positive rates
    all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        mean_precision += np.interp(all_recall, recall[i], precision[i])
    # Finally average it and compute AUC
    mean_precision /= n_classes
    recall["macro"] = all_recall[::-1]
    precision["macro"] = mean_precision
    pr_auc["macro"] = auc(recall["macro"], precision["macro"])

    plt.step(recall["micro"], precision["micro"],
            label='micro-average ROC curve (AUC = {0:0.4f})'
                ''.format(pr_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)
    
    plt.step(recall["macro"], precision["macro"],
            label='macro-average ROC curve (AUC = {0:0.4f})'
                ''.format(pr_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)


    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    #plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.title('Precision-Recall curve ({})'.format(title), fontsize= 14)
    plt.savefig(save_path, dpi=600)