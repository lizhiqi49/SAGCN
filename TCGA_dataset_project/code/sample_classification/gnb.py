
from nested_cv_func import *
from sklearn.naive_bayes import GaussianNB


#===== load dataset
random_sorted_sample_df = pd.read_csv('../../data/sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(np.int)
X = pd.read_csv('../../data/meth_matrix_probes_150.csv', index_col=0).loc[:,random_sorted_sample].values.T.astype(float)
print("data loaded")

#===== perform 5x5 nested cv

## GaussianNB
param_space_gnb = {
    "var_smoothing": [1e-9]
}
model_gnb = GaussianNB()
print("========GaussianNB nested cv start===========")
print_time()
nested_cv(X, y, model_gnb, param_space_gnb, './result_gnb.csv')