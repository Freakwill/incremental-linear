#!/usr/bin/env python3

"""Incremental Bayesian Linear Regression

Incremental Learning based on Bayesian Linear Regression

*Reference* 
Fletcher T. Relevance Vector Machines Explained 2010. http://home.mit.bme.hu/~horvath/IDA/RVM.pdf

"""

import numpy as np
import numpy.linalg as LA
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel

EPSILON = 1e-16

def _eap(gram, design_y, y_norm_squre, N, p, init_alpha=1, init_sigma_square=1, n_iter=100):
    """Evidence Approximation Procedure
    initialize alpha beta
    loop:
        E step:
        Sigma_ = (sigma_square + np.diag(alpha) gram)^-1
        Sigma__ = sigma_square Sigma_
        Sigma = Sigma__ np.diag(alpha)
        mu = Sigma_ (alpha o design_y)
        M step:
        1. gamma_i=1-alpha_i Sigma_ii
        2. alpha_i= gamma_i/ mu_i^2
        3. beta = (N- sum_i gamma_i) / ||t- Phi  mu ||^2
    output alpha beta, stor mu, sigma
    """

    if isinstance(init_alpha, (float, int)):
        init_alpha *= np.ones(p)
    gram, design_y, y_norm_squre = gram /N, design_y /N, y_norm_squre /N
    alpha = init_alpha
    sigma_square = init_sigma_square
    last_alpha = alpha
    for _ in range(n_iter):
        # E step
        Sigma_ = LA.inv(sigma_square * np.eye(p) + np.diag(alpha) @ gram)
        Sigma__ = sigma_square * Sigma_
        Sigma = Sigma__ @ np.diag(alpha)
        mu = np.dot(Sigma_, alpha * design_y)

        # M step
        gamma = 1 - np.diag(Sigma__)
        alpha = N*np.array([0 if g < EPSILON else m**2 / g for g, m in zip(gamma, mu)])
        sigma_square = (np.dot(np.dot(gram, mu), mu)-2*np.dot(mu, design_y)+y_norm_squre) / (1 - gamma.sum()/N)
        if sigma_square < EPSILON or LA.norm(alpha-last_alpha, ord=np.inf) < EPSILON:
            break
        last_alpha = alpha
    return alpha, sigma_square, mu, Sigma


def _ieap(gram, design_y, y_norm_squre, r, p, sigma_square=1, mu=0, Sigma=None):
    # Incremental Version of Evidence Approximation Procedure
    # do not update alpha
    if sigma_square < EPSILON:
        return sigma_square, mu, Sigma
    Sigma_ = LA.inv(sigma_square * np.eye(p) + Sigma @ gram)
    mu = Sigma_ @ (np.dot(Sigma, design_y) + sigma_square * mu)
    Sigma = sigma_square * Sigma_ @ Sigma
    sigma_square = (1-r) * sigma_square + r * (np.dot(np.dot(gram, mu), mu)-2*np.dot(mu, design_y)+y_norm_squre)
    return sigma_square, mu, Sigma


class IncrementalLinearRegression(RegressorMixin, LinearModel):
    """Incremental Bayesian Linear Regression

    Y | w, sigma^2 ~ Xw + N(0, sigma^2), w|alpha ~ N(0, alpha)

    Note that var of w denoted by alpha^{-1} in formal literatures.

    calculate alpha and sigma^2 with Evidence Approximation Procedure
    
    Extends:
        RegressorMixin
        LinearModel
    """
    def __init__(self, init_alpha=1, init_sigma_square=1, warm_start=False):
        """
        Keyword Arguments:
            init_alpha {number|array} -- init value for hyper paramter in priori dist. of N (default: {1})
            init_sigma_square {number} -- i.v. for variance (default: {1})
            warm_start {bool} -- flag for incremental learning
        """
        self.init_alpha = init_alpha
        self.init_sigma_square = init_sigma_square
        self.warm_start = warm_start
        self.flag = False
        self.__features = None

    @property
    def features(self):
        return self.__features
    

    def fit(self, X, y):
        if self.warm_start and self.flag:
            self.partial_fit(X, y)
        else:
            self.init(X, y)
            self.alpha, self.sigma_square, self.mu, self.Sigma = _eap(
                self.gram, self.design_y, self.y_norm_squre, self.n_observants, self.n_features, 
                self.init_alpha, self.init_sigma_square)
            self.coef_ = self.mu[:-1]
            self.intercept_ = self.mu[-1]
            self.postprocess()
        return self


    def partial_fit(self, X, y, r=None):
        if not self.flag:
            raise Exception('There is no initial value for alpha or sigma_square!')
        self.init(X, y, warm_start=True)
        self.sigma_square, self.mu, self.Sigma = _ieap(
            self.gram, self.design_y, self.y_norm_squre, (r or self.r_observants), self.n_features, 
            self.sigma_square, self.mu, self.Sigma)
        self.coef_ = self.mu[:-1]
        self.intercept_ = self.mu[-1]
        self.postprocess()
        return self

    def design_matrix(self, X):
        N, p = X.shape
        if hasattr(X, 'columns'):
            features = tuple(X.columns) + ('常数项',)
        else:
            features = np.arange(p+1)
        return np.hstack((X, np.ones((N, 1)))), features

    def init(self, X, y, warm_start=False):
        # get information of normal equation
        self.flag = True
        design, features = self.design_matrix(X)
        n_observants, n_features = design.shape

        if warm_start:
            if n_features != self.n_features:
                # X = self.preprocess(X)
                raise Exception('Number of features should be keep constant.')
            self.gram = design.T @ design
            self.design_y = np.dot(design.T, y)
            self.y_norm_squre = np.dot(y, y)
            self.n_observants += n_observants
            self.r_observants = n_observants / self.n_observants
            # self.n_observants += n_observants
            # self.gram = gram * p + self.gram * (1-p)
            # self.design_y = design_y *p + self.design_y * (1-p)
            # self.y_norm_squre = y_norm_squre *p + self.y_norm_squre * (1-p)
        else:
            self.__features = features
            self.n_observants = n_observants
            self.n_features = n_features
            self.gram = design.T @ design
            self.design_y = np.dot(design.T, y)
            self.y_norm_squre = np.dot(y, y)

    def important_features(self, threshold=None):
        if threshold:
            ind = self.alpha > threshold
        else:
            ind = self.alpha > 0
        if self.features is None:
            return tuple([i for i, k in enumerate(ind) if k])
        else:
            return tuple([self.features[i] for i, k in enumerate(ind) if k])

    def remove_dispensable(self, threshold=None):
        original_features = self.features
        self.__features = self.important_features(threshold=threshold)
        ind = [k for k, f in enumerate(original_features) if f in self.features]
        self.gram = self.gram[ind, ind]
        self.alpha = self.alpha[ind]
        self.Sigma = self.Sigma[ind, ind]
        self.design_y = self.design_y[ind]


    def postprocess(self):
        pass



if __name__ == '__main__':

    print('receive data')
    X=np.array([[1,2,1],[3,3,2], [4,5,3],[5,6,4]])
    y=np.array([3, 6,9.5,10.5])
    print('create a model (set warm_start=True)')
    a = IncrementalLinearRegression(warm_start=True)
    a.fit(X, y)
    print(f'''
    coef: {a.coef_}
    training score: {a.score(X, y)}
    ''')

    print('save the model')
    import joblib
    joblib.dump(a, 'a.model')
    print('load the model')
    a= joblib.load('a.model')
    print('receive new data')
    print(f'''previous coef: {a.coef_}
        flag: {a.flag} (if False, then partial_fit will raise an error!)''')
    X=np.array([[5,6,10],[4,3,2], [4,7,6],[5,8,10]])
    y=np.array([11, 8,11,13])
    a.fit(X, y)
    print(f'''
    coef: {a.coef_}
    training score: {a.score(X, y)}
    important features: {a.important_features(0.001)}
    ''')
    

