from nested_cv_func import *
from sklearn.ensemble import ExtraTreesClassifier


#===== load dataset
random_sorted_sample_df = pd.read_csv('../../data/sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(np.int)
X = pd.read_csv('../../data/meth_matrix_probes_150.csv', index_col=0).loc[:,random_sorted_sample].values.T.astype(float)
print("data loaded")

#===== perform 5x5 nested cv

## extraTreesClassifier
param_space_ert = {
    "n_estimators": list(range(50,300,50)),  
    "min_samples_split": list(range(2,6,1))
}
model_ert = ExtraTreesClassifier()
print("========ExtraTreesClassifier nested cv start===========")
print_time()
nested_cv(X, y, model_ert, param_space_ert, './result_ert.csv')