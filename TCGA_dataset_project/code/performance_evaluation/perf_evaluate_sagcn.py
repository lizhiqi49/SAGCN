#%%
# performance evaluation of SAGCN for different ks


from metrics_calculate import *

def performance_evaluation_sagcn():
   
    num_sites = [50, 100, 150, 200, 250]
    for site in num_sites:
        base_dir = 'G:/MyGit/SAGCN/TCGA_dataset_project/prediction_results/sagcn_results/'
        output_dir = 'G:/MyGit/SAGCN/TCGA_dataset_project/perf_evaluation_results/sagcn/'
        proba_file = "probas_sagcn_{}sites.csv".format(str(site))
        proba_df = pd.read_csv(base_dir+proba_file, index_col=0)
        probas_nocalibrated = proba_df.values[:,:12]
        probas_calibrated = proba_df.values[:, 12:-1].astype(float)
        y_real = proba_df.values[:, -1].astype(int)

        
        metric_file_nocalibrated = "metrics_sagcn_{}sites_nocali.csv".format(str(site))
        computeMetrics(y_real, probas_nocalibrated, output_dir + metric_file_nocalibrated)
        metric_file_calibrated = "metrics_sagcn_{}sites_calibrated.csv".format(str(site))
        computeMetrics(y_real, probas_calibrated, output_dir + metric_file_calibrated)
"""
        roc_file_nocali = "roc_sagcn_{}sites_nocali.png".format(str(site))
        roc_title_nocali = 'Receiver operating characteristic (SAGCN: k3={:.1f}, no calibrated)'.format(site/500)
        drawROC(probas_nocalibrated, y_real, 12, roc_title_nocali, output_dir + roc_file_nocali)

        roc_file_calibrated = "roc_sagcn_{}sites_calibrated.png".format(str(site))
        roc_title_calibrated = 'Receiver operating characteristic (SAGCN: k3={:.1f}, calibrated)'.format(site/500)
        drawROC(probas_nocalibrated, y_real, 12, roc_title_calibrated, output_dir + roc_file_calibrated)

        pr_file_nocali = "pr_sagcn_{}sites_nocali.png".format(str(site))
        pr_title_nocali = "Precision-Recall curve (SAGCN: k3={:.1f}, no calibrated)".format(site/500)
        drawPR(probas_nocalibrated, y_real, 12, pr_title_nocali, output_dir + pr_file_nocali)

        pr_file_calibrated = "pr_sagcn_{}sites_calibrated.png".format(str(site))
        pr_title_calibrated = "Precision-Recall curve (SAGCN: k3={:.1f}, calibrated)".format(site/500)
        drawPR(probas_calibrated, y_real, 12, pr_title_calibrated, output_dir + pr_file_calibrated)
"""
if __name__ == '__main__':
    performance_evaluation_sagcn()
# %%
