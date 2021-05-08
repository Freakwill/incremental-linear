#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import numpy as np
from sklearn.base import RegressorMixin

from multiprocessing import Process, Manager, set_start_method
        

class MORegression(RegressorMixin):
    def __init__(self, models=None, model=None):
        if models and isinstance(models, (list, tuple)):
            self.models = models
            self.model = None
        elif model:
            self.model = model
            self.models = None
        else:
            raise Exception('You have to supply keyword argument `models` or `model`')

        if models:
            self.__ndim_ouput = len(models)
        else:
            self.__ndim_ouput = 0

    @property
    def ndim_ouput(self):
        return self.__ndim_ouput

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, m):
        self.__model = m

    @property
    def models(self):
        return self.__models

    @models.setter
    def models(self, m):
        self.__models = m
    
        
    def fit(self, X, Y, *args, **kwargs):
        import pandas as pd
        self.ndim_output = Y.shape[1]
        if not self.models and self.__model:
            self.models = [copy.deepcopy(self.__model) for _ in range(self.ndim_output)]
        
        if not isinstance(Y, np.ndarray):
            if isinstance(Y, pd.DataFrame):
                Y = Y.values
            else:
                Y = np.array(Y)
        for model, y in zip(self.models, Y.T):
            model.fit(X, y)

        # set_start_method('fork')
        # manager = Manager()
        # results = manager.list()

        # def target(model, x, y, results): 
        #     results.append(model.fit(x, y))

        # processes = [Process(target=target, args=(model, X, y, []), *args, **kwargs)
        #     for model, y in zip(self.models, Y.T)]
        # for process in processes:
        #     process.start()
        # for process in processes:
        #     process.join()
        return self

    def predict(self, X):
        return np.array([model.predict(X) for model in self.models]).T

    def get_params(self, *args, **kwargs):
        d = {}
        for model in self.models:
            d.update(model.get_params(*args, **kwargs))
        return d

    def __getitem__(self, k):
        return self.models[k]

    @classmethod
    def fromRegressor(cls, reg, n=1):
        # copy a single regressor
        mor = cls(models=[copy.deepcopy(reg) for _  in range(n)])
        mor.__model = reg
        return mor


    def score(self, X, Y):
        return np.mean([model.score(X, y) for model, y in zip(self.models, Y.T)])
