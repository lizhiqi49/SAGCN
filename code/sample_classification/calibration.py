#%%

# lzq
# calibration
# 2021-09-25

# ----------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def onehotCoding(arr, num_classes):
    return np.eye(num_classes)[arr]


class CalibrationModel(object):
    def __init__(self):
        self.logistic_reg = LogisticRegression(multi_class='ovr')

    def fit(self, y_probas, y_real):
        self.logistic_reg.fit(y_probas, y_real)

    def calibrate(self, y_probas):
        return self.logistic_reg.predict_proba(y_probas)


# %%
