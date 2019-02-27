# -*- coding: utf-8 -*-
"""
Just a benchmark test for the PredictFutureSales Problem. We will use the sales of the previous month and see the score.
"""

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import xgboost as xgb
import time

np.set_printoptions(precision=3, suppress=True, threshold=np.nan)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:2.2f}'.format)
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30

data_dir = os.path.join(os.getcwd(), 'data')

### Load the last data
dict_types_v3 = {'date_block_num':np.int8, 'shop_id':np.int8, 'item_category':np.int8, 'item_id': np.int16, 'item_price':np.float32, 'item_cnt_day':np.int16, 'day':np.int8, 'month':np.int8, 'year': np.int16}
df_train = pd.read_csv(os.path.join(data_dir, 'sales_train_v3.csv'), dtype = dict_types_v3)
df_test = pd.read_csv(os.path.join(data_dir, 'test_v4.csv'), dtype = dict_types_v3)

# Extract features and Aggregate by (shop_id, item_id, month), i.e., for a shop, item and month sum all the sales during that month
baseline_features = ['shop_id', 'item_id', 'item_category', 'date_block_num', 'item_cnt_day']
df_train['item_cnt_day'] = df_train['item_cnt_day'].clip(0,100)
df_train = df_train[baseline_features].groupby(['shop_id', 'item_id', 'date_block_num']).agg({'item_cnt_day':np.sum, 'item_category':np.mean}).reset_index() # I have to be careful here if item_cat are not all the same (although they SHOULD be, and they are)
df_train.rename(columns={'item_cnt_day':'item_cnt_month'}, inplace=True)

baseline_features.remove('item_cnt_day')
baseline_features.append('item_cnt_month')

# Remove pandas index column
df_train['item_cnt_month'] = df_train.item_cnt_month.fillna(0).clip(0,100)

df_trainx = df_train.drop('item_cnt_month', axis=1)
df_trainy = df_train['item_cnt_month']
print(df_trainx.columns)

# Make test_dataset pandas data frame, add category id and date block num, then convert back to numpy array and predict
df_test['date_block_num'] = 33
print(df_test.head())

print('Creating the model')
model = xgb.XGBRegressor(max_depth =1, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 10, seed = 1)
print('Model created, now fitting.')
model.fit(df_trainx.values, df_trainy.values, eval_metric='rmse')
print('Model fit on data. Now predicting')
preds = model.predict(df_test[df_trainx.columns].values)
print('Prediction done.')

#df = pd.DataFrame(preds, columns = ['item_cnt_month'])
df_test['item_cnt_month'] = preds
df_test[['ID', 'item_cnt_month']].to_csv('simple_xgb.csv')





