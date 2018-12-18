import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
#from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV   #Perforing grid search

train = pd.read_csv("train_all.csv")
dtest = pd.read_csv("republish_test.csv")

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
    dtest[key] = pd.to_numeric(dtest[key], errors='coerce').fillna(0, downcast='infer')

le = LabelEncoder()
train_X = train[feature_columns]
le.fit(train["current_service"])
train_y = le.transform(train['current_service'])
train_y = pd.DataFrame({'current_service': train_y})

# train_y = train["current_service"]
# test_X = test[feature_columns].values

# train_X, validate_X, test_X = np.split(train_X.sample(frac=1), [int(.6 * len(train_X)), int(.8 * len(train_X))])
# train_y, validate_y, test_y = np.split(train_y.sample(frac=1), [int(.6 * len(train_X)), int(.8 * len(train_y))])

train_X, test_X, train_y, test_y = \
    train_test_split(train_X, train_y, test_size=0.2, random_state=0)
valid_X, test_X, valid_y, test_y = \
    train_test_split(test_X, test_y, test_size=0.5, random_state=0)
print "training a XGBoost classifier\n"

params = {
    'max_depth': 9, 'silent': 1, 'eta': 0.05, 'seed': 2018, 'gamma': 0,
    'max_delta_step': 0, 'subsample': 1, 'alpha': 1, 'lambda': 1, 'scale_pos_weight': 1,
    'n_estimators': 802, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9, 'base_score': 0.5,
    'objective': 'multi:softmax', 'num_class': 15
}

num_rounds = 3000
xgtrain = xgb.DMatrix(train_X, label=train_y, feature_names=feature_columns)
xgval = xgb.DMatrix(test_X, label=test_y, feature_names=feature_columns)
# xgtest = xgb.DMatrix(test_X, label=test_y, feature_names=feature_columns)

watchlist = [(xgtrain, 'train'), (xgval, 'eval')]
bst = xgb.train(params, xgtrain, num_rounds, watchlist) # , early_stopping_rounds=10
# bst.fit(xgtrain, label, eval_set=[(train[feature], label)], verbose=1, )
# get prediction
valid_X = xgb.DMatrix(valid_X)
pred = bst.predict(valid_X)

valid_y = valid_y['current_service'].tolist()
error_rate = np.sum(pred != valid_y) / (len(valid_y) * 1.0)
print('Test error using softmax = {}'.format(error_rate))

# feat_imp = pd.Series(bst.get_booster().get_score(importance_type='weight')).sort_values(ascending=False)
# feat_imp.plot(kind='bar', title='Feature Importances')
# plt.ylabel('Feature Importance Score')
#from matplotlib import pyplot
#xgb.plot_importance(bst, title="Feature Importantance")
#pyplot.show()
# plt.show()

pred = bst.predict(xgb.DMatrix(dtest[feature_columns]))
print('[INFO] test predicted.')

print pred
pred = le.inverse_transform([int(x) for x in pred])
dtest['current_service'] = pred
dtest[['user_id', 'current_service']].to_csv('./xgboost_0930.csv', index=False)

pickle.dump(bst, open("xgb-model-0930", "wb"))
