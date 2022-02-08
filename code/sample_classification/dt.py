import sys
from cv_func import *
from sklearn.tree import DecisionTreeClassifier


#===== load dataset
num_probe = int(sys.argv[1])
random_sorted_sample_df = pd.read_csv('../../data/sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(np.int)
X = pd.read_csv('../../data/meth_matrix_probes_{}.csv'.format(str(num_probe)), index_col=0).loc[:,random_sorted_sample].values.T.astype(float)
print(X.shape)
print("data loaded")


model_dt = DecisionTreeClassifier(criterion='gini', min_samples_split=3)
print("========DecisionTreeClassifier cv start===========")
print_time()
cross_validation(X, y, model_dt, './result_dt_{}probes.csv'.format(str(num_probe)))
print_time()