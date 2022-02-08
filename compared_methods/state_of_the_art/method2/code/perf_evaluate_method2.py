#%%

# performance evaluation of state-of-the-art method(1)
# 2021.10.09
# ----------------------------------------------------------------

import os
from metrics_calculate import *

proba_file = "../prediction_results/probas_method_2.csv"
output_dir = "../perf_evaluation_results/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
proba_df = pd.read_csv(proba_file, index_col=0)
probas = proba_df.values[:, :12]
y_real = proba_df.values[:, -1].astype(int)
metric_file = "metrics_state_of_the_art_method_2.csv"
computeMetrics(y_real, probas, output_dir+metric_file)

roc_file = "roc_state_of_the_art_method_2.png"
roc_title = "Receiver operating characteristic (SOTA method 2)"
drawROC(probas, y_real, 12, roc_title, output_dir+roc_file)

pr_file = "pr_state_of_the_art_method_2.png"
pr_title = "Precision-Recall curve (SOTA method 2)"
drawPR(probas, y_real, 12, pr_title, output_dir+pr_file)