#!/usr/bin/env python3

"""
An example of incremental linear model
"""

from utils import *
import pandas as pd
import numpy as np

# prepare the data
X_keys = ['小缸面料大类', '小缸光谱', '小缸配方组合', '小缸浓度']
Y_keys = ['大缸物料代码', '大缸浓度']
keys = X_keys + Y_keys

try:
    data = pd.read_csv('data.csv')
except:
    data = pd.read_excel('original-data.xlsx', sheet_name=0, index_col=0)
    data = data[keys]

    data = split(data, ['大缸光谱'])
    data = dual(data, ['大缸物料代码', '小缸配方组合'], ['大缸浓度', '小缸浓度'])
    data.to_csv('data.csv')

X_keys = [c for c in data.columns if any(c.startswith(k) for k in X_keys)]
Y_keys = [c for c in data.columns if c.startswith(Y_keys[0])]
X, Y = data[X_keys], data[Y_keys].values

# build the model and test it
from sklearn.model_selection import train_test_split
from models import models
from mo import MORegression
from sklearn.linear_model import *
from incremental_linear import *
import time

columns = ('旧数据训练分数', '旧数据测试分数', '新数据训练分数', '新数据测试分数', '第一次测试分数', '第二次测试分数', '第一次训练耗时/s', '第二次训练耗时/s')

scores = np.empty(8)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_old, X_new, Y_old, Y_new = train_test_split(X_train, Y_train, test_size=0.4)
model = MORegression(model=IncrementalLinearRegression(warm_start=True))  # You have to set warm_start=True
time_start = time.perf_counter()
model.fit(X_old, Y_old)
time_end = time.perf_counter()
time1 = time_end - time_start
scores[0] = model.score(X_old, Y_old)
scores[3] = model.score(X_new, Y_new)
scores[4] = model.score(X_test, Y_test)
time_start = time.perf_counter()
model.fit(X_new, Y_new)
time_end = time.perf_counter()
time2 = time_end - time_start
scores[1] = model.score(X_old, Y_old)
scores[2] = model.score(X_new, Y_new)
scores[5] = model.score(X_test, Y_test)
scores[6] = time1
scores[7] = time2

# print results
for c, s in zip(columns, scores):
    print(f'{c}: {s:.4}')

