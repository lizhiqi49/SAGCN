import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from calibration import CalibrationModel


def print_time():
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def onehotCoding(arr, num_classes):
    return np.eye(num_classes)[arr]

def nested_cv(X, y, algorithm, param_space, predict_proba_file, fold=5, scoring='f1_macro'):
    r"""
    params:
        X: features of all samples
        y: labels of all samples
        algorithm: the classifier algorithm used
        param_space: the input of grid search
        predict_proba_file: the output path of predict probas
        fold: default 5
        calibration: Ture or False
        scoring: scoring metric
    """

    inner_cv = KFold(n_splits=fold, shuffle=False)
    outer_cv = KFold(n_splits=fold, shuffle=False)
    
    scoring_metric = scoring
    #==== 5x5 nested cv
    fold = 0
    for train_ind, test_ind in outer_cv.split(y):
        fold += 1
        print("outer cv fold: %d" % fold)
        grid_search = GridSearchCV(estimator=algorithm, param_grid=param_space, 
                                scoring=scoring_metric, cv=inner_cv)
        grid_search.fit(X[train_ind], y[train_ind])

        print("best_params:", grid_search.best_params_)
        print("best_score:", grid_search.best_score_)

        test_label = y[test_ind].reshape((-1,1))
        test_proba = grid_search.predict_proba(X[test_ind])

        train_proba = grid_search.predict_proba(X[train_ind])
        calibration_model = CalibrationModel()
        calibration_model.fit(train_proba, y[train_ind])
        test_proba_calibrated = calibration_model.calibrate(test_proba)
        
        result = np.hstack((test_proba, test_proba_calibrated))
        result = np.hstack((result, test_label))

        if fold == 1:
            results = result
        else:
            results = np.vstack((results, result))

         
    results = pd.DataFrame(results)
    results.to_csv(predict_proba_file)