
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import pickle
# File system manangement
import os
from sklearn import preprocessing

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import cross_validation

train = pd.read_csv("train_all.csv")
test = pd.read_csv("republish_test.csv")

# print train['current_service'].value_counts()
# print train['current_service'].value_counts().plot.hist()

feature_columns = [
    "service_type", "is_mix_service", "online_time", "1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee",
    "month_traffic", "many_over_bill", "contract_type", "contract_time", "is_promise_low_consume", "net_service",
    "pay_times", "pay_num", "last_month_traffic", "local_trafffic_month", "local_caller_time", "service1_caller_time",
    "service2_caller_time", "gender", "age", "complaint_level", "former_complaint_num", "former_complaint_fee",
]
#     "current_service", "user_id"

keys = ["2_total_fee", "3_total_fee", "gender", "age"]
for key in keys:
    test[key] = pd.to_numeric(test[key], errors='coerce').fillna(0, downcast='infer')

le = LabelEncoder()
label = le.fit(train["current_service"])


if __name__ == "__main__":
    clf = pickle.load(open("xgb-model-0928-afternoon", "rb"))

    pred = clf.predict(xgb.DMatrix(test[feature_columns]))
    print('[INFO] test predicted.')

    print pred
    pred = le.inverse_transform([int(x) for x in pred])
    test['current_service'] = pred
    test[['user_id', 'current_service']].to_csv('./xgboost_0928.csv', index=False)