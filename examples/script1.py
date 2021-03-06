#!/usr/bin/env python3

"""Numerical Experiment for Incremental Bayesian Linear Regression

Data are generated by program not obtained from industry.
"""

from utils import *
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from models import *
from sklearn.linear_model import *
from incremental_linear import *
import time

# generate data
from sklearn.datasets import make_regression
n_features = 100
noise_old, noise_new, noise_test = 100, 75, 50

# the names of models
columns = ('旧数据训练分数', '旧数据测试分数', '新数据训练分数', '新数据测试分数', '第一次测试分数', '第二次测试分数', '第一次训练耗时/s', '第二次训练耗时/s')
model_names =('增量Bayes线性回归', '只学习新数据', '一次性学习', 'SGD回归', 'MLP回归', '普通线性回归', 'Bayes脊回归')

# test
n_trials = 2
scores = np.empty((7,8, n_trials))

for _ in range(n_trials):
    print(f'Trial {_} starts...')

    print('generating data...')
    X_old, Y_old, c_old = make_regression(n_samples=5000, n_features=n_features, n_informative=20, bias=10, noise=noise_old, coef=True)
    X_new, Y_new, c_new = make_regression(n_samples=1000, n_features=n_features, n_informative=20, bias=10, noise=noise_new, coef=True)
    X_new += np.random.randn(1000, n_features) * 5
    # About 50% coefs are selected to be perturbated
    c_new = c_old + np.random.randn(n_features) * (np.random.random(n_features)<0.5)
    Y_new = np.dot(X_new, c_new) + 10 + np.random.randn(1000) * noise_new
    X_train, Y_train = np.vstack((X_old, X_new)), np.hstack((Y_old, Y_new))
    X_test, Y_test, c_test = make_regression(n_samples=1000, n_features=n_features, n_informative=20, bias=10, noise=noise_test, coef=True)
    c_test = c_new + np.random.randn(n_features) * 0.5 * (np.random.random(n_features)<0.5)
    X_test += np.random.randn(1000, n_features) * 10
    Y_test = np.dot(X_test, c_test) + 10 + np.random.randn(1000) * noise_test

    # my method
    print('My algorithm of incremental linear')
    model = IncrementalLinearRegression(warm_start=True)
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

    # only learn new data
    print('My algorithm, but only learn new data')
    model = IncrementalLinearRegression()
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
    print('My algorithm, but learn old and new data one time in the second stage')
    model = IncrementalLinearRegression()
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

    # SGD Regressor
    print('SGD Regression')
    model = model=SGDRegressor(warm_start=True, max_iter=2000)
    time_start = time.perf_counter()
    model.fit(X_old, Y_old)
    time_end = time.perf_counter()
    time1 = time_end - time_start
    scores[3, 0, _] = model.score(X_old, Y_old)
    scores[3, 3, _] = model.score(X_new, Y_new)
    scores[3, 4, _] = model.score(X_test, Y_test)
    time_start = time.perf_counter()
    model.fit(X_new, Y_new)
    time_end = time.perf_counter()
    time2 = time_end - time_start
    scores[3, 1, _] = model.score(X_old, Y_old)
    scores[3, 2, _] = model.score(X_new, Y_new)
    scores[3, 5, _] = model.score(X_test, Y_test)
    scores[3, 6, _] = time1
    scores[3, 7, _] = time2

    # MLP Regressor
    print('MLP Regression')
    model = model=MLPRegressor(warm_start=True, max_iter=2000)
    time_start = time.perf_counter()
    model.fit(X_old, Y_old)
    time_end = time.perf_counter()
    time1 = time_end - time_start
    scores[4, 0, _] = model.score(X_old, Y_old)
    scores[4, 3, _] = model.score(X_new, Y_new)
    scores[4, 4, _] = model.score(X_test, Y_test)
    time_start = time.perf_counter()
    model.fit(X_new, Y_new)
    time_end = time.perf_counter()
    time2 = time_end - time_start
    scores[4, 1, _] = model.score(X_old, Y_old)
    scores[4, 2, _] = model.score(X_new, Y_new)
    scores[4, 5, _] = model.score(X_test, Y_test)
    scores[4, 6, _] = time1
    scores[4, 7, _] = time2

    # learn old and new data one time with common linear model
    print('Ordinary Linear Regression')
    model = LinearRegression()
    time_start = time.perf_counter()
    model.fit(X_old, Y_old)
    time_end = time.perf_counter()
    time1 = time_end - time_start
    scores[5, 0, _] = model.score(X_old, Y_old)
    scores[5, 3, _] = model.score(X_new, Y_new)
    scores[5, 4, _] = model.score(X_test, Y_test)
    time_start = time.perf_counter()
    model.fit(X_train, Y_train)
    time_end = time.perf_counter()
    time2 = time_end - time_start
    scores[5, 1, _] = model.score(X_old, Y_old)
    scores[5, 2, _] = model.score(X_new, Y_new)
    scores[5, 5, _] = model.score(X_test, Y_test)
    scores[5, 6, _] = time1
    scores[5, 7, _] = time2

    # learn old and new data one time with common linear model
    print('Bayesian Ridge Regression (not incremental)')
    model = model=BayesianRidge()
    time_start = time.perf_counter()
    model.fit(X_old, Y_old)
    time_end = time.perf_counter()
    time1 = time_end - time_start
    scores[6, 0, _] = model.score(X_old, Y_old)
    scores[6, 3, _] = model.score(X_new, Y_new)
    scores[6, 4, _] = model.score(X_test, Y_test)
    time_start = time.perf_counter()
    model.fit(X_train, Y_train)
    time_end = time.perf_counter()
    time2 = time_end - time_start
    scores[6, 1, _] = model.score(X_old, Y_old)
    scores[6, 2, _] = model.score(X_new, Y_new)
    scores[6, 5, _] = model.score(X_test, Y_test)
    scores[6, 6, _] = time1
    scores[6, 7, _] = time2

scores = np.median(scores, axis=2)
scores = pd.DataFrame(data=scores, columns=columns, index=model_names)
def convert(x):
    if np.isscalar(x):
        return f"{x:.4}"
    else:
        return str(x)

scores=scores.applymap(convert)
print(scores)
