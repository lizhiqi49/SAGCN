
# train svc model based on TCGA dataset
# 2021.10.29
from cv_func import print_time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import joblib


#===== load dataset
random_sorted_sample_df = pd.read_csv('../../data/sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(np.int)
X = pd.read_csv('../../data/meth_matrix_probes_350.csv', index_col=0).loc[:,random_sorted_sample].values.T.astype(float)
print_time()
print("data loaded")

model = SVC(C=1, gamma=0.01, kernel='rbf', probability=True)
model.fit(X, y)
joblib.dump(model, "model_svc_tcga_350.pkl")
print_time()
print("finish training")
