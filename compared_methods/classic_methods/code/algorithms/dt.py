#%%

from nested_cv_func import *
from sklearn.tree import DecisionTreeClassifier

#%%

#===== load dataset
random_sorted_sample_df = pd.read_csv('../../data/sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(np.int)
X = pd.read_csv('../../meth_matrix_probes_150.csv', index_col=0).loc[:,random_sorted_sample].values.T.astype(float)
print("data loaded")

#%%

#===== perform 5x5 nested cv on decision tree



param_space_dt = {
    "criterion": ['gini', 'entropy'],
    "min_samples_split": list(range(2,6,1))
}
model_dt = DecisionTreeClassifier()
print("========DecisionTreeClassifier nested cv start===========")
print_time()
nested_cv(X, y, model_dt, param_space_dt, './result_dt.csv')