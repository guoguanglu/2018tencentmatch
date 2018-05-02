# coding=utf-8
# @author:guoguanglu
# github: https://github.com/guoguanglu/2018tencentmatch
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                       'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
result = pd.merge(left, right, on=['key1'],how='left')
s=[1,239,2,48.0,9,21]
le=LabelEncoder()
y=right[['C']]
print y
print le.fit_transform(s)
