#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import *
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import *

from tpot import TPOTRegressor

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense




class MyKernel(Matern):
    """classifying-kernel
    """
    scale = 15
    def __call__(self, X, Y=None, eval_gradient=False):
        X = X.copy()
        X[:, 6:] *=MyKernel.scale
        if Y is not None:
            Y = Y.copy()
            Y[:, 6:] *=MyKernel.scale
        return super(MyKernel, self).__call__(X, Y=Y, eval_gradient=eval_gradient)

    # scale = lambda x: 20*(1-x)

    # def __call__(self, X, Y=None, eval_gradient=False):
    #     X1, C= X[:,:6], X[:, 6:]
    #     if Y is None:
    #         H = pdist(C, metric='hamming')
    #         H = squareform(MyKernel.scale(H))
    #         np.fill_diagonal(H, 1)
    #         Y1=None
    #     else:
    #         Y1, D= Y[:,:6], Y[:, 6:]
    #         H = cdist(C, D, metric='hamming')
    #         H = MyKernel.scale(H)
    #     if eval_gradient:
    #         K, g = super(MyKernel, self).__call__(X1, Y=Y1, eval_gradient=eval_gradient)
    #         return K+H, g
    #     else:
    #         K = super(MyKernel, self).__call__(X1, Y=Y1, eval_gradient=eval_gradient)
    #         return K+H


def create_model():

    model = Sequential()
    model.add(Dense(300, input_dim=361, kernel_initializer='uniform', activation='relu'))
    #model.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def scaling(scale):
    def func(X):
        X = X.copy()
        X[:, 6:] *= scale
        return X
    return func

class MyTransformer(FunctionTransformer):
    def __init__(self, scale=30, *args, **kwargs):
        super(MyTransformer, self).__init__(func=scaling(scale=scale), *args, **kwargs)


models={
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=0.01),
    'Ransac': RANSACRegressor(),
    'Bayes': BayesianRidge(),
    'Lasso': LassoLarsCV(cv=5),
    # 'Huber': HuberRegressor(),
    # 'KR': KernelRidge(alpha=0.01, kernel=MyKernel()),
    # 'LinearSVR': LinearSVR(),
    # 'SVR': SVR(kernel=MyKernel()),
    # 'GP': GaussianProcessRegressor(alpha=0.03, n_restarts_optimizer=5, kernel=Matern()),
    # 'MyGP': GaussianProcessRegressor(alpha=0.03, n_restarts_optimizer=5, kernel=MyKernel()),
    # 'MLP': MLPRegressor(max_iter=5000, hidden_layer_sizes=(50,), solver='lbfgs',learning_rate='adaptive'),
    # 'TPOT': TPOTRegressor(generations=10, population_size=25),
    # 'GBoost': GradientBoostingRegressor(alpha=0.85, learning_rate=0.12, loss="ls",
    #                                           max_features=0.4, min_samples_leaf=5,
    #                                           min_samples_split=6),
    # 'Ada': AdaBoostRegressor(LassoLarsCV(cv=5), n_estimators=10),
    # 'Adax':AdaBoostRegressor(GaussianProcessRegressor(alpha=0.02, n_restarts_optimizer=5, kernel=MyKernel())),
}

