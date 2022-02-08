import sys
from cv_func import *
from sklearn.ensemble import ExtraTreesClassifier


#===== load dataset
num_probe = int(sys.argv[1])
random_sorted_sample_df = pd.read_csv('../../data/sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(np.int)
X = pd.read_csv('../../data/meth_matrix_probes_{}.csv'.format(str(num_probe)), index_col=0).loc[:,random_sorted_sample].values.T.astype(float)
print(X.shape)
print("data loaded")

## random forest
"""
param_space_rf = {
    "n_estimators": list(range(200,500,50)),  
    "min_samples_split": [2,3,4,5]
}
model_rf = RandomForestClassifier()
"""
model_rf = ExtraTreesClassifier(n_estimators=300, min_samples_split=3)
print("========Random Forest nested cv start===========")
print_time()
#nested_cv(X, y, model_rf, './result_rf_250probes.csv')
cross_validation(X, y, model_rf, 'result_ert_{}probes.csv'.format(str(num_probe)))
print("done")
print_time()