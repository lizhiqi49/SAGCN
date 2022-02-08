#%%
#
# performance evaluation of sample classification on different algorithms

from metrics_calculate import *

def performance_evaluation_sample_classify():
    #algorithms = ['dt', 'gnb', 'rf', 'ert']
    #num_probes = [50, 100, 150, 200, 250, 300, 350]
    #num_probes = [250]
    algorithms = ['svc']
    base_dir = 'G:/MyGit/SAGCN/TCGA_dataset_project/prediction_results/clinical_sample_classification/'
    output_dir = 'G:/MyGit/SAGCN/TCGA_dataset_project/perf_evaluation_results/clinical_sample_classification/'
    for algorithm in algorithms:
        proba_file = "result_{}_350probes.csv".format(algorithm)
        #proba_file = "probas_gnn_sc.csv"
        proba_df = pd.read_csv(base_dir + proba_file, index_col=0)
        probas_nocalibrated = proba_df.values[:, :12]
        #probas_calibrated = proba_df.values[:, 12:24]
        y_real = proba_df.values[:, -1].astype(int)
        #metric_file_nocalibrated = "metrics_svc_{}probes_.csv".format(str(num_probe))
        #metric_file_nocalibrated = "metrics_{}_350probes.csv".format(algorithm)
        #computeMetrics(y_real, probas_nocalibrated, output_dir+metric_file_nocalibrated)
        #metirc_file_calibrated = "metrics_{}_calibrated_250sites.csv".format(algorithm)
        #computeMetrics(y_real, probas_calibrated, output_dir+metirc_file_calibrated)


        roc_file_nocalibrated = "roc_{}_350probes.png".format(algorithm)
        roc_title_nocalibrated = "Receiver operating characteristic (sample classification on 350 probes)"
        #roc_file_calibrated = "roc_{}_calibrated.png".format(algorithm)
        #roc_title_calibrated = "Receiver operating characteristic ({}_calibrated)".format(algorithm)
        drawROC(probas_nocalibrated, y_real, 12, roc_title_nocalibrated, output_dir+roc_file_nocalibrated)
        #drawROC(probas_calibrated, y_real, 12, roc_title_calibrated, output_dir+roc_file_calibrated)

        pr_file_nocalibrated = "pr_{}_350probes.png".format(algorithm)
        pr_title_nocalibrated = "Precision-Recall curve (sample classification on 350 probes)"
        #pr_file_calibrated = "pr_{}_calibrated.png".format(algorithm)
        #pr_title_calibrated = "Precision-Recall curve ({}_calibrated)".format(algorithm)
        drawPR(probas_nocalibrated, y_real, 12, pr_title_nocalibrated, output_dir+pr_file_nocalibrated)
        #drawPR(probas_calibrated, y_real, 12, pr_title_calibrated, output_dir+pr_file_calibrated)



if __name__ == "__main__":
    performance_evaluation_sample_classify()
# %%
