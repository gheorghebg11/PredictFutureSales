# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:28:16 2019
@author: Bogdan
"""

import numpy as np
import pandas as pd
import os

np.set_printoptions(precision=3, suppress=True, threshold=np.nan)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.1f}'.format)

## Load the Data
data_dir = os.path.join(os.getcwd(), 'data')

df_train = pd.read_csv(os.path.join(data_dir, 'sales_train_v2.csv'))
df_items_and_cats = pd.read_csv(os.path.join(data_dir, 'items-translated.csv'))

## Some global variables
nbr_days = df_train['date'].value_counts().shape[0]


## Some Functions
def fix_returned_items(df):
	days_with_returns = df[df['item_cnt_day'] < 0]['date']
	nbr_days_with_returns = days_with_returns.value_counts().shape[0]
	print(f'There are {nbr_days_with_returns} days with returns out of {nbr_days}.')
	print(f'There are {days_with_returns.shape[0]} returned transactions out of {df_train.index.shape[0]}.')

	study_returned_items_values = False
	if study_returned_items_values:
		sold_vs_returns = []
		for day in days_with_returns.unique():
			df_day_neg = df[(df['date'] == day) & (df['item_cnt_day'] < 0)]
			df_day_pos = df[(df['date'] == day) & (df['item_cnt_day'] > 0)]
			sold_vs_returns.append([df_day_pos['item_price'].dot(df_day_pos['item_cnt_day']), df_day_neg['item_price'].dot(df_day_neg['item_cnt_day'])])
			print([df_day_pos['item_price'].dot(df_day_pos['item_cnt_day']), df_day_neg['item_price'].dot(df_day_neg['item_cnt_day'])])

	remove_returned_items_rows = True
	if remove_returned_items_rows:
		df.drop(df[df['item_cnt_day'] < 0].index, inplace=True)
		df.reset_index(drop=True, inplace=True)
		print(f'Removed rows when items returned')

def fix_negative_prices(df):
	days_with_neg_prices = df[df['item_price'] < 0]['item_price']
	nbr_days_with_neg_prices = days_with_neg_prices.shape[0]
	print(f'There are {nbr_days_with_neg_prices} items with negative prices out of {df.shape[0]}.')

	remove_item_with_neg_price = True
	if remove_item_with_neg_price:
		df.drop(df[df['item_price'] < 0].index, inplace=True)
		df.reset_index(drop=True, inplace=True)
		print(f'Removed rows when item has negative price')

def add_item_id_to_train(df, dict_of_items):

	df['category_id'] = df['item_id'].map(dict_of_items)


## Execute the Preprocessing / Visualization
study_items_and_cats = True
if study_items_and_cats:
	print(df_items_and_cats.info())
	print(df_items_and_cats.describe())

study_training_set = True
if study_training_set:
	fix_returned_items(df_train)
	fix_negative_prices(df_train)

	dict_of_items = dict(zip(df_items_and_cats['item_id'], df_items_and_cats['item_category_id']))
	add_item_id_to_train(df_train, dict_of_items)

	print(df_train.info())
	print(df_train.describe())







