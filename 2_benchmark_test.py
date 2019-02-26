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


### Set the next sales as the sales from the previous month
last_month = df_train['date_block_num'].max()

df_train_last = df_train.query('date_block_num == @last_month')[['shop_id', 'item_id','item_cnt_day']]
df_train_last['item_cnt_day'] = df_train_last['item_cnt_day'].clip(0,20)
df_train_last = df_train_last.groupby(['shop_id', 'item_id'])['item_cnt_day'].agg(np.sum).reset_index()
df_train_last.rename(columns={'item_cnt_day': 'prediction'}, inplace=True)

df_test['item_cnt_month'] = df_test.merge(df_train_last, how='left', on=['shop_id', 'item_id'])['prediction'].fillna(0).clip(0,20)

submission = df_test[['ID','item_cnt_month']]

submission.to_csv(os.path.join(data_dir,'benchmark_prev_month.csv'), index=False)



