import copy
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from calibration import CalibrationModel


def print_time():
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def onehotCoding(arr, num_classes):
    return np.eye(num_classes)[arr]
    
    
def cross_validation(X, y, model, predict_proba_file, fold=5):
    cv = KFold(n_splits=fold, shuffle=False)
    for fold, (train_ind, test_ind) in enumerate(cv.split(y)):
        print("cv fold: %d" % fold)
        model_cv = copy.deepcopy(model)
        model_cv.fit(X[train_ind], y[train_ind])
        
        test_label = y[test_ind].reshape((-1,1))
        test_proba = model_cv.predict_proba(X[test_ind])
        
        result = np.hstack((test_proba, test_label))

        if fold == 0:
            results = result
        else:
            results = np.vstack((results, result))
            
    results = pd.DataFrame(results)
    results.to_csv(predict_proba_file)
    

def nested_cv(X, y, model, predict_proba_file, fold=5):
    r"""
    params:
        X: features of all samples
        y: labels of all samples
        model: the originally defined model
        predict_proba_file: the output path of predict probas
        fold: default 5
    """

    inner_cv = KFold(n_splits=fold, shuffle=False)
    outer_cv = KFold(n_splits=fold, shuffle=False)
    
    #==== 5x5 nested cv
    # outer loop
    for outer_fold, (train_ind_outer, test_ind_outer) in enumerate(outer_cv.split(y)):
        print("====================================")
        print("outer cv fold: %d" % outer_fold)
        y_inner = y[train_ind_outer]
        X_inner = X[train_ind_outer]

        # inner loop
        for inner_fold, (train_ind_inner, test_ind_inner) in enumerate(inner_cv.split(y_inner)):
            print("inner cv fold: %d" % inner_fold)
            model_inner = copy.deepcopy(model)
            model_inner.fit(X_inner[train_ind_inner], y_inner[train_ind_inner])
            test_proba_inner = model_inner.predict_proba(X[test_ind_inner])
            if inner_fold == 0:
                test_proba_inner_all = test_proba_inner
                y_test_inner_all = y_inner[test_ind_inner]
            else:
                test_proba_inner_all = np.vstack((test_proba_inner_all, test_proba_inner))
                y_test_inner_all = np.concatenate((y_test_inner_all, y_inner[test_ind_inner]))
            print(test_proba_inner_all.shape)
            del model_inner


        model_outer = copy.deepcopy(model)
        model_outer.fit(X[train_ind_outer], y[train_ind_outer])
        test_proba_outer = model_outer.predict_proba(X[test_ind_outer])

        # train calibration model
        calibration_model = CalibrationModel()
        calibration_model.fit(test_proba_inner_all, y_test_inner_all)
        test_proba_outer_calibrated = calibration_model.calibrate(test_proba_outer)

        result = np.hstack((test_proba_outer, test_proba_outer_calibrated))
        result = np.hstack((result, y[test_ind_outer].reshape((-1, 1))))

        if outer_fold == 0:
            results = result
        else:
            results = np.vstack((results, result))
        del model_outer

         
    results = pd.DataFrame(results)
    results.to_csv(predict_proba_file)


def nested_cv_old(X, y, algorithm, param_space, predict_proba_file, fold=5, scoring='f1_macro'):
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