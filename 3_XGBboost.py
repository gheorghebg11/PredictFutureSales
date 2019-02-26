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

# Remove pandas index column
df_train['item_cnt_month'] = df_train.item_cnt_month.fillna(0).clip(0,100)


print(df_test.head())


# Make test_dataset pandas data frame, add category id and date block num, then convert back to numpy array and predict
merged_test['date_block_num'] = 33
merged_test.set_index('shop_id')
merged_test.head(3)

model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, subsample = 1, eta = 0.3, num_round = 1000, seed = 1)
model.fit(trainx, trainy, eval_metric='rmse')
preds = model.predict(merged_test.values)

df = pd.DataFrame(preds, columns = ['item_cnt_month'])
df['ID'] = df.index
df = df.set_index('ID')
df.to_csv('simple_xgb.csv')



