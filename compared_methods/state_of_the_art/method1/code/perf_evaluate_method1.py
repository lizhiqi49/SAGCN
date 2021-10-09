#%%

# performance evaluation of state-of-the-art method(1)
# 2021.10.09
# ----------------------------------------------------------------

from metrics_calculate import *

proba_file = "../prediction_results/probas_method_1.csv"
output_dir = "../perf_evaluation_results/"
proba_df = pd.read_csv(proba_file, index_col=0)
probas = proba_df.values[:, :12]
y_real = proba_df.values[:, -1].astype(int)
metric_file = "metrics_state_of_the_art_method_1.csv"
computeMetrics(y_real, probas, output_dir+metric_file)

roc_file = "roc_state_of_the_art_method_1.png"
roc_title = "Receiver operating characteristic (STOA method 1)"
drawROC(probas, y_real, 12, roc_title, output_dir+roc_file)

pr_file = "pr_state_of_the_art_method_1.png"
pr_title = "Precision-Recall curve (STOA method 1)"
drawPR(probas, y_real, 12, pr_title, output_dir+pr_file)