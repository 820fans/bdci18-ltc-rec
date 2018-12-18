
# coding: utf-8

# In[4]:

import pickle
import pandas as pd

train = pd.read_csv("train_all.csv")
test = pd.read_csv("republish_test.csv")


# In[5]:


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

drop_columns = [
    "former_complaint_num", "net_service", "former_complaint_fee", "is_mix_service", "service_type"
]
train.drop(drop_columns)
test.drop(drop_columns)
# In[6]:


# encode label
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train['current_service'])
train.loc[:,'service_no'] = pd.Series(le.transform(train['current_service']), index=train.index)
train.head()


# In[7]:


# 简短的weight
from sklearn.utils import class_weight
import numpy as np
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train['service_no']),
                                                 train['service_no'])
class_weights


# In[13]:


train_y = train['service_no'].values

train_X = train[feature_columns].values
test_X = test[feature_columns].values


# In[22]:


import sklearn.metrics as metrics
def micro_avg_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='micro')

y_target = np.zeros((test.shape[0], N))


# In[23]:


from sklearn.model_selection import StratifiedKFold

N = 11
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=1684).split(train_X, train_y)


# In[24]:


import xgboost as xgb
dtest = xgb.DMatrix(test_X)
param = {'max_depth': 9, 'eta':0.05, 'eval_metric':'merror', 
             'max_delta_step': 0, 'subsample': 1, 'alpha': 1, 'lambda': 1, 'scale_pos_weight': 1,
    'n_estimators': 802, 'colsample_bytree': 0.9, 'colsample_bylevel': 0.9, 
         'silent':0, 'objective':'multi:softmax', 'num_class':11, 'seed': 2006}  # 参数


# In[25]:


vcc = 0
for i ,(train_fold,test_fold) in enumerate(skf):
    
    X_train, X_validate, y_train, y_validate =        train_X[train_fold, :], train_X[test_fold, :], train_y[train_fold], train_y[test_fold]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalidate = xgb.DMatrix(X_validate, label=y_validate)
    
    # 直接xgb分类
    evallist  = [(dtrain,'train'), (dvalidate,'validate')] 
    num_round = 200  # 循环次数
    clf = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=20)
    
    model_name = "model1014/xgb-kf-model-1014-round %d" % i
    pickle.dump(clf, open(model_name, "wb"))
    val_1 = clf.predict(dvalidate)
    vcc += micro_avg_f1(y_validate, val_1)
    result = clf.predict(dtest)
    y_target[:, i] = result

print(vcc/N)


# In[26]:


from collections import Counter
service_type = []
for i in range(y_target.shape[0]):
    tmp = []
    for j in range(N):
        tmp.append(int(y_target[i][j]))
    word_counts = Counter(tmp)
    yes = word_counts.most_common(1)
    service_type.append(le.inverse_transform(yes[0][0]))
service_type


# In[28]:


submit = pd.DataFrame(columns=['user_id'])
submit['user_id']=test['user_id']
submit['current_service'] = pd.Series(service_type, index=target.index)
submit.to_csv("ltc-kf-xgb.csv", index=False)


# In[ ]:




