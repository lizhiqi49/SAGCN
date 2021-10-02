from nested_cv_func import *
from sklearn.linear_model import LogisticRegression


#===== load dataset
random_sorted_sample_df = pd.read_csv('../../data/sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(np.int)
X = pd.read_csv('../../data/meth_matrix_probes_150.csv', index_col=0).loc[:,random_sorted_sample].values.T.astype(float)
print("data loaded")

#===== perform 5x5 nested cv

## ElasticNet
param_space_elnet = {
    "penalty": ['elasticnet'], 
    "C": np.logspace(-3, 1, 8, base=2),
    "solver": ['saga'],
    "multi_class": ['auto'],
    "l1_ratio": np.arange(0.1, 1, 0.2),
    "max_iter": [400]
}
model_elnet = LogisticRegression()
print("========ElasticNet nested cv start===========")
print_time()
nested_cv(X, y, model_elnet, param_space_elnet, './result_elnet.csv')