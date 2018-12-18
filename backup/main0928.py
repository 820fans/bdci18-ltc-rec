import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

import pickle
# File system manangement
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt

train = pd.read_csv("train_all.csv")
test  = pd.read_csv("republish_test.csv")

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
    train[key] = pd.to_numeric(train[key], errors='coerce').fillna(0, downcast='infer')
    test[key] = pd.to_numeric(test[key], errors='coerce').fillna(0, downcast='infer')

train_X = train[feature_columns].values
train_y = train["current_service"]
test_X = test[feature_columns].values

# print train.dtypes.value_counts()
# print test.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# train[] = pd.to_numeric(train_X, errors='coerce').fillna(0, downcast='infer')
# print train_X
# exit()

gbm = xgb.XGBClassifier(max_depth=9, n_estimators=792, learning_rate=0.05,
gamma=0, max_delta_step=0, subsample=1, colsample_bytree=0.9, colsample_bylevel=0.9,
                            reg_alpha=1, reg_lambda=1, scale_pos_weight=1,
                            base_score=0.5, seed=2018,
                        objective='multi:softmax', num_class=15)

# silent=True,
gbm.fit(train_X, train_y)
predictions = gbm.predict(test_X)

from xgboost import plot_importance
from matplotlib import pyplot


submission = pd.DataFrame({'user_id': test['user_id'], 'current_service': predictions})
# print(submission)
submission.to_csv("submission1002.csv", index=False)


# save model to file
#pickle.dump(gbm, open("xgb-model-0928-afternoon", "wb"))
#pickle.load("xgbmodel/xgb-model-0928-afternoon")