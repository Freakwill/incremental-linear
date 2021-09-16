#!/usr/bin/env python3

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

from sklearn.model_selection import train_test_split
from models import models
from mo import MORegression
from sklearn.linear_model import *
from incremental_linear import *
import time

columns = ('旧数据训练分数', '旧数据测试分数', '新数据训练分数', '新数据测试分数', '第一次测试分数', '第二次测试分数', '第一次训练耗时/s', '第二次训练耗时/s')
model_names =('增量Bayes线性回归', '只学习新数据', '一次性学习', '普通线性回归', 'Bayes脊回归')

n_trials = 5
scores = np.empty((5,8, n_trials))

for _ in range(1, n_trials+1):
    print(f'Trial {_}/{n_trials} starts...')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_old, X_new, Y_old, Y_new = train_test_split(X_train, Y_train, test_size=0.4)
    # my method
    model = MORegression(model=IncrementalLinearRegression(warm_start=True))
    time_start = time.perf_counter()
    model.fit(X_old, Y_old)
    time_end = time.perf_counter()
    time1 = time_end - time_start
    scores[0, 0, _] = model.score(X_old, Y_old)
    scores[0, 3, _] = model.score(X_new, Y_new)
    scores[0, 4, _] = model.score(X_test, Y_test)
    time_start = time.perf_counter()
    model.fit(X_new, Y_new)
    time_end = time.perf_counter()
    time2 = time_end - time_start
    scores[0, 1, _] = model.score(X_old, Y_old)
    scores[0, 2, _] = model.score(X_new, Y_new)
    scores[0, 5, _] = model.score(X_test, Y_test)
    scores[0, 6, _] = time1
    scores[0, 7, _] = time2

    # only learn old data
    # model = MORegression(model=IncrementalLinearRegression())
    # time_start = time.perf_counter()
    # model.fit(X_old, Y_old)
    # time_end = time.perf_counter()
    # time1 = time_end - time_start
    # scores[1, 0, _] = model.score(X_old, Y_old)
    # scores[1, 1, _] = scores[1, 2, _] = np.nan  # no data
    # scores[1, 3, _] = model.score(X_new, Y_new)
    # scores[1, 4, _] = model.score(X_test, Y_test)
    # scores[1, 5, _] = np.nan
    # scores[1, 6, _] = time1
    # scores[1, 7, _] = np.nan

    # only learn new data
    model = MORegression(model=IncrementalLinearRegression(warm_start=False))
    time_start = time.perf_counter()
    model.fit(X_new, Y_new)
    time_end = time.perf_counter()
    time1 = time_end - time_start
    scores[1, 1, _] = model.score(X_old, Y_old)
    scores[1, 2, _] = model.score(X_new, Y_new)
    scores[1, 0, _] = scores[1, 3, _] = np.nan
    scores[1, 4, _] = np.nan
    scores[1, 5, _] = model.score(X_test, Y_test)
    scores[1, 6, _] = np.nan
    scores[1, 7, _] = time1

    # learn old and new data one time
    time_start = time.perf_counter()
    model.fit(X_old, Y_old)
    time_end = time.perf_counter()
    time1 = time_end - time_start
    scores[2, 0, _] = model.score(X_old, Y_old)
    scores[2, 3, _] = model.score(X_new, Y_new)
    scores[2, 4, _] = model.score(X_test, Y_test)
    time_start = time.perf_counter()
    model.fit(X_train, Y_train)
    time_end = time.perf_counter()
    time2 = time_end - time_start
    scores[2, 1, _] = model.score(X_old, Y_old)
    scores[2, 2, _] = model.score(X_new, Y_new)
    scores[2, 5, _] = model.score(X_test, Y_test)
    scores[2, 6, _] = time1
    scores[2, 7, _] = time2

    # learn old and new data one time with common linear model
    model = LinearRegression()
    time_start = time.perf_counter()
    model.fit(X_old, Y_old)
    time_end = time.perf_counter()
    time1 = time_end - time_start
    scores[3, 0, _] = model.score(X_old, Y_old)
    scores[3, 3, _] = model.score(X_new, Y_new)
    scores[3, 4, _] = model.score(X_test, Y_test)
    time_start = time.perf_counter()
    model.fit(X_train, Y_train)
    time_end = time.perf_counter()
    time2 = time_end - time_start
    scores[3, 1, _] = model.score(X_old, Y_old)
    scores[3, 2, _] = model.score(X_new, Y_new)
    scores[3, 5, _] = model.score(X_test, Y_test)
    scores[3, 6, _] = time1
    scores[3, 7, _] = time2

    # learn old and new data one time with common linear model
    model = MORegression(model=BayesianRidge())
    time_start = time.perf_counter()
    model.fit(X_old, Y_old)
    time_end = time.perf_counter()
    time1 = time_end - time_start
    scores[4, 0, _] = model.score(X_old, Y_old)
    scores[4, 3, _] = model.score(X_new, Y_new)
    scores[4, 4, _] = model.score(X_test, Y_test)
    time_start = time.perf_counter()
    model.fit(X_train, Y_train)
    time_end = time.perf_counter()
    time2 = time_end - time_start
    scores[4, 1, _] = model.score(X_old, Y_old)
    scores[4, 2, _] = model.score(X_new, Y_new)
    scores[4, 5, _] = model.score(X_test, Y_test)
    scores[4, 6, _] = time1
    scores[4, 7, _] = time2

scores = np.median(scores, axis=2)
scores = pd.DataFrame(data=scores, columns=columns, index=model_names)
print(scores)
# scores.to_csv('scores2.csv')
