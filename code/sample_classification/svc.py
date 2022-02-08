from nested_cv_func import *
from sklearn.svm import SVC
import sys

num_probe = int(sys.argv[1])

#===== load dataset
random_sorted_sample_df = pd.read_csv('../../data/sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(np.int)
X = pd.read_csv('../../data/meth_matrix_probes_{}.csv'.format(str(num_probe)), index_col=0).loc[:,random_sorted_sample].values.T.astype(float)
print(X.shape)
print("data loaded")

#===== perform 5x5 nested cv
## svc
"""
param_space_svc = {
    "C": np.logspace(-2, 1, 5, base=2),
    "kernel": ['rbf'],
    "gamma": [0.01, 0.1] + ['auto'],
    "probability": [True]
}
model_svc = SVC()
"""
model_svc = SVC(C=1, kernel='rbf', gamma=0.01, probability=True)
print("========SVC nested cv start===========")
print_time()
nested_cv(X, y, model_svc, './result_svc_{}probes.csv'.format(str(num_probe)))
#nested_cv(X, y, model_svc, param_space_svc, './result_svc_250probes.csv')
print("done")
print_time()