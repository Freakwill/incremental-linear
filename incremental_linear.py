#!/usr/bin/env python3

"""Incremental Bayesian Linear Regression

Incremental Learning based on Bayesian Linear Regression

Toy Example:

    print('receive data')
    X=np.array([[1,2,1],[3,3,2], [4,5,3],[5,6,4]])
    y=np.array([3, 6,9.5,10.5])
    print('create a model (set warm_start=True)')
    ilr = IncrementalLinearRegression(warm_start=True)
    ilr.fit(X, y)
    print(f'''
    coef: {ilr.coef_}
    training score: {ilr.score(X, y)}
    ''')

    print('Saving the model')
    import joblib
    joblib.dump(ilr, 'ilr.model')
    print('Saved the model')
    print('Loading the model')
    ilr = joblib.load('ilr.model')
    print('Receive new data')
    print(f'''
        previous coef: {ilr.coef_}
        ''')
    X = np.array([[5,6,10],[4,3,2], [4,7,6],[5,8,10]])
    y = np.array([11, 8,11,13])
    ilr.fit(X, y)
    print(f'''After incremental learning.
    coef: {ilr.coef_}
    training score: {ilr.score(X, y)}
    informative features: {ilr.informative_features(0.001)}
    ''')

    print('Create a transfer learning model from `ilr`')
    tlr = TransferLinearRegression.from_model(transfered_model=ilr)
    X = np.array([[5,6,10],[4,3,2], [4,7,6],[5,8,10]])
    y = np.array([11, 8,11,13])
    tlr.fit(X, y)
    print(f'''After Transfer learning.
    coef: {tlr.coef_}
    training score: {tlr.score(X, y)}
    informative features: {tlr.informative_features(0.001)}
    ''')

Numerical Experiment:

    see files in the folder `examples/*`


*Reference* 
Fletcher T. Relevance Vector Machines Explained 2010. http://home.mit.bme.hu/~horvath/IDA/RVM.pdf

"""

import numpy as np
import numpy.linalg as LA
from sklearn.base import RegressorMixin
from sklearn.linear_model._base import LinearModel
from sklearn.utils.validation import check_is_fitted

__version__ = '3.0'

EPSILON = 1e-16

def scalar_matrix(*args, **kwargs):
    # scalar matrix
    return np.diag(np.full(*args, **kwargs))

def _design_matrix(X, fit_intercept=True):
    """Make design matrix
    
    For ordinary linear square, it would be X.T@X, ie the Gram matrix
    Override it, if you need.
    
    Arguments:
        X {2D array} -- input data
        fit_intercept {bool} -- whether to add intercept to the design matrix 
    
    Returns:
        design matrix and names/indexes of features
    """
    N, p = X.shape
    if fit_intercept:
        if hasattr(X, 'columns'):
            if 'intercept' in self.columns:
                raise Exception('It seems that the design matrix has contained `intercept`, please check it.')
            features = tuple(X.columns) + ('intercept',)
        else:
            features = np.arange(p+1)
        return np.hstack((X, np.ones((N, 1)))), features
    else:
        if hasattr(X, 'columns'):
            features = tuple(X.columns)
        else:
            features = np.arange(p)
        return X, features

def feature_check(features1, features2):
    # check the equality of two tuples of features
    if features1 is not None:
        if len(features1) != len(features2):
            raise Exception("""Features of data and the model should be keep identical.
                but they are not equal to each other on length.
                """)
        if any(f1!=f2 for f1, f2 in zip(features1, features2)):
            raise Exception("""Features of data and the model should be keep identical.
                It is recommanded to set self.feature = None, or amend data of X.
                """)

def _find(feature, features):
    # find a `feature` from `features` list
    for k, f in enumerate(features):
        if f == feature:
            return k
    return -1

def _bool_index(ind, features=None):
    # boolean index for `features` by `ind`
    if features is None:
        return tuple(np.nonzero(ind)[0])
    else:
        return tuple(features[i] for i, k in enumerate(ind) if k)


def _eap(gram, design_y, y_norm_squre, N, p, init_alpha=1, init_sigma_square=1, n_iter=100):
    """Evidence Approximation Procedure

    initialize alpha beta
    loop:
        E step:
        Sigma_ = (sigma_square + diag(alpha) * gram)^-1
        Sigma__ = sigma_square * Sigma_
        Sigma = Sigma__ * diag(alpha)
        mu = Sigma_ (alpha .* design_y)
        M step:
        1. gamma_i = 1 - alpha_i * Sigma_ii
        2. alpha_i = gamma_i ./ mu_i^2
        3. beta = (N- sum_i gamma_i) / ||t- Phi  mu ||^2
    output alpha, beta; store mu, sigma
    """

    if np.isscalar(init_alpha):
        init_alpha *= np.ones(p)
    gram, design_y, y_norm_squre = gram /N, design_y /N, y_norm_squre /N
    alpha = init_alpha
    sigma_square = init_sigma_square
    last_alpha = alpha
    for _ in range(n_iter):
        # E step
        Sigma_ = LA.inv(scalar_matrix(p, sigma_square) + np.diag(alpha) @ gram)
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
    """Incremental Version of Evidence Approximation Procedure
    mainly update mu, Sigma, but do not update alpha

    Sigma_ <- (sigma^2 + Sigma gram)
    mu <- Sigma_ * ((Sigma * design_y) + sigma_square mu)
    Sigma <- sigma_square * Sigma_ * Sigma
    """
    if sigma_square < EPSILON:
        return sigma_square, mu, Sigma

    A = scalar_matrix(p, sigma_square) + Sigma @ gram
    X = LA.solve(A, np.column_stack((Sigma, mu)))
    X1, X2 = X[:,:-1], X[:,-1]
    mu = np.dot(X1, design_y) + sigma_square * X2
    Sigma = sigma_square * X1

    # old code
    # Sigma_ = LA.inv(scalar_matrix(p, sigma_square) + Sigma @ gram)
    # mu = Sigma_ @ (np.dot(Sigma, design_y) + sigma_square * mu)
    # Sigma = sigma_square * Sigma_ @ Sigma
    
    sigma_square_ = np.dot(np.dot(gram, mu), mu) + y_norm_squre - 2*np.dot(mu, design_y)
    sigma_square += r * (sigma_square_ - sigma_square)

    return sigma_square, mu, Sigma

def _teap(gram, design_y, y_norm_squre, p, sigma_square=1, mu=0, Sigma=None):
    """Transfer learning version of _ieap

    just let r = 1 in _ieap
    """
    if sigma_square < EPSILON:
        return sigma_square, mu, Sigma
    
    A = scalar_matrix(p, sigma_square) + Sigma @ gram
    X = LA.solve(A, np.column_stack((Sigma, mu)))
    X1, X2 = X[:,:-1], X[:,-1]
    mu = np.dot(X1, design_y) + sigma_square * X2
    Sigma = sigma_square * X1

    sigma_square = np.dot(np.dot(gram, mu), mu) + y_norm_squre - 2*np.dot(mu, design_y)

    return sigma_square, mu, Sigma


def _normal_equation(design, y, N=1):
    """Make normal equation
    """
    gram = design.T @ design
    design_y = np.dot(design.T, y)
    y_norm_squre = np.dot(y, y)
    return gram / N, design_y / N, y_norm_squre / N


class BaseIncrementalLinearRegression(RegressorMixin, LinearModel):

    def __init__(self, rate=None, warm_start=True, fit_intercept=True, max_iter=100):
        """
        Keyword Arguments:
            rate {number} -- the weight of new data (default: {None}, proportion of new data in whole train data)
            warm_start {bool} -- flag to execute incremental learning, if False, it would not do incremental learning (default: {True})
            max_iter {number} -- maximum of iterations (only used in the first fitting step)
            fit_intercept {bool} -- fit the intercept of the linear model or not (default: {True})
        """
        self.warm_start = warm_start
        self.rate = rate
        self.__flag = False
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self._features = None

    @property
    def features(self):
        return self._features

    @property
    def n_features(self):
        if self.features is None:
            return 0
        else:
            return len(self.features)

    @features.setter
    def features(self, v):
        if self._features is None:
            self._features = v
        elif len(self._features) < len(v):
            self.filter_features(v)
        elif v is None:
            self._features = None
        else:
            raise NotImplementedError('''
        Sorry, renaming features with longer length is not implemented currently.
        It is suggested that users filtrate the features of the model by the method `filter_features` before
        incremental learning.
        ''')

    def fit(self, X, y):
        if self.warm_start and self.__flag:
            self.partial_fit(X, y)
        else:
            self._fit(X, y)
        return self


    def _init(self, X, y, warm_start=False):
        # Get information of normal equation
        design, features = _design_matrix(X, fit_intercept=self.fit_intercept)
        n_observants = design.shape[0]

        feature_check(self.features, features)

        if warm_start:
            self.n_observants_ += n_observants
            if self.rate is None:
                self.rate = n_observants / self.n_observants_
        else:
            self._features = features
            self.n_observants_ = n_observants
            self.__flag = True
         
        self.gram_, self.design_y_, self.y_norm_squre_ = _normal_equation(design, y)


    def set_coef(self):
        if self.fit_intercept:
            self.coef_ = self.mu_[:-1]
            self.intercept_ = self.mu_[-1]
        else:
            self.coef_ = self.mu_
            self.intercept_ = 0


    def transfer_fit(self, X, y):
        raise NotImplementedError(f"No transfer learning method for {self.__class__.__name__}")


class IncrementalGramianLinearRegression(BaseIncrementalLinearRegression):
    """Incremental liear regression in a naive way.
    """

    def _init(self, X, y, warm_start=False):
        # Get information of normal equation
        design, features = _design_matrix(X, fit_intercept=self.fit_intercept)
        n_observants = design.shape[0]

        feature_check(self.features, features)

        if warm_start:
            self.n_observants_ += n_observants
            if self.rate is None:
                r = self.rate = n_observants / self.n_observants_
            gram, design_y, y_norm_squre = _normal_equation(design, y, N=n_observants)
            self.gram_ = r * gram + (1 - r) * self.gram_
            self.design_y_ = r * design_y + (1 - r) * self.design_y_
            self.y_norm_squre_ = r * y_norm_squre + (1 - r) * self.y_norm_squre_
        else:
            self._features = features
            self.n_observants_ = n_observants
            self.__flag = True
            self.gram_, self.design_y_, self.y_norm_squre_ = _normal_equation(design, y, N=n_observants)


    def _fit(self, X, y):
        # Call EAP to compute the parameters
        self._init(X, y)
        self.mu_ = LA.solve(self.gram_, self.design_y_)
        self.sigma_square_ = np.dot(np.dot(self.gram_, self.mu_), self.mu_) + self.y_norm_squre_ - 2*np.dot(self.mu_, self.design_y_)
        self.set_coef()


    def partial_fit(self, X, y, rate=None):
        # Do incremental learning by Bayesian method
        if not self.__flag:
            raise Exception('The model may not be initalized sufficiantly!')
        check_is_fitted(self, ('sigma_square_', 'mu_', 'Sigma_'))

        gram = self.gram_
        design_y = self.design_y_

        self._init(X, y, warm_start=True)
        self.mu_ = LA.solve(self.gram_, self.design_y_)

        self.set_coef()
        return self

class IncrementalBayesianLinearRegression(BaseIncrementalLinearRegression):
    """Incremental Bayesian Linear Regression

    It is based on the following linear model:
    Y | sigma^2 ~ Xw + N(0, sigma^2), w|alpha ~ N(0, alpha)
    where X is the design matrix, w, alpha and sigma^2 are unknown parameters.
    
    Note that var of w denoted by alpha^{-1} in formal literatures.

    Idea:
    1. calculate alpha and sigma^2 with Evidence Approximation Procedure
    2. update mu and Sigma in one step, from that on.

    You May need to store the model with following codes:
        ```
        import joblib
        joblib.dump(il, 'il.model')
        il= joblib.load('il.model')
        ```
    
    Extends:
        RegressorMixin, LinearModel
    """

    def __init__(self, init_alpha=1, init_sigma_square=1, **params):
        """
        Keyword Arguments:
            init_alpha {number|array} -- initial value for hyper paramter in priori distribution of N (default: {1})
            init_sigma_square {number} -- initial value for variance (default: {1})
            rate {number} -- the weight of new data (default: {None}, proportion of new data in whole train data)
            warm_start {bool} -- flag to execute incremental learning, if False, it would not do incremental learning (default: {True})
            max_iter {number} -- maximum of iterations (only used in the first fitting step)
        """
        super().__init__(**params)
        self.init_alpha = init_alpha
        self.init_sigma_square = init_sigma_square


    def _fit(self, X, y):
        # Call EAP to compute the parameters
        self._init(X, y)

        # Use attribute `alpha_` and `sigma_square_`, if they are available.
        alpha = getattr(self, 'alpha_', self.init_alpha)
        sigma_square = getattr(self, 'sigma_square_', self.init_sigma_square)

        self.alpha_, self.sigma_square_, self.mu_, self.Sigma_ = _eap(
            self.gram_, self.design_y_, self.y_norm_squre_, self.n_observants_, self.n_features, 
            alpha, sigma_square, self.max_iter)
        self.set_coef()


    def partial_fit(self, X, y, rate=None):
        # Do incremental learning by Bayesian method
        if not self.__flag:
            raise Exception('The model may not be initalized sufficiantly!')
        check_is_fitted(self, ('sigma_square_', 'mu_', 'Sigma_'))

        self._init(X, y, warm_start=True)

        self.sigma_square_, self.mu_, self.Sigma_ = _ieap(
            self.gram_, self.design_y_, self.y_norm_squre_, (rate or self.rate), self.n_features, 
            self.sigma_square_, self.mu_, self.Sigma_)
        self.set_coef()
        return self


    def transfer_fit(self, X, y):
        """For transfer learning

        It is identical with partial_fit(self, X, y, rate=1)
        setting rate=1 to forget the old observants.
        """

        check_is_fitted(self, ('sigma_square_', 'mu_', 'Sigma_'))

        self._init(X, y, warm_start=False)

        self.sigma_square_, self.mu_, self.Sigma_ = _teap(
            self.gram_, self.design_y_, self.y_norm_squre_, self.n_features, 
            self.sigma_square_, self.mu_, self.Sigma_)
        self.set_coef()
        return self


    def _init(self, X, y, warm_start=False):
        # Get information of normal equation
        design, features = _design_matrix(X, fit_intercept=self.fit_intercept)
        n_observants = design.shape[0]

        feature_check(self.features, features)
 
        self.gram_ = design.T @ design
        self.design_y_ = np.dot(design.T, y)
        self.y_norm_squre_ = np.dot(y, y)

        if warm_start:
            self.n_observants_ += n_observants
            if self.rate is None:
                self.rate = n_observants / self.n_observants_
        else:
            self._features = features
            self.n_observants_ = n_observants
            self.__flag = True


    def informative_features(self, threshold=0):
        """Get informative features whose weights are greater then threshold

        Arguments:
            threshold {number} -- the threshold that weights of
                                  informative features should be greater than

        Return:
            tuple of informative features
        """

        ind = self.alpha_ > threshold # or np.isclose(self.mu_, 0)
        return _bool_index(ind, self.features)


    def remove_dispensable(self, threshold=None):
        # Remove non-informative features
        self.filter_features(self.informative_features(threshold=threshold))
        

    def filter_features(self, features):
        """Features should be contained in self.features

        Filter the (informative) features from the original by the feature names or the indexes;
        The parameters should be updated accordingly.

        It is recommanded to use the data type of DataFrame for input variables,
        if you want to use this method to change features.

        Arguments:
            features --- subset of self.features
        """
        
        ind = []
        for feature in features:
            i = _find(feature, self.features)
            if i != -1:
                ind.append(i)
        self.gram_ = self.design_[ind, ind]
        self.alpha_ = self.alpha_[ind]
        self.Sigma_ = self.Sigma_[ind, ind]
        self.design_y_ = self.design_y_[ind]
        self._features = features


class TransferBayesianLinearRegression(IncrementalBayesianLinearRegression):
    # Please use the static method `from_model`
    # to create a TransferBayesianLinearRegression object
    
    @classmethod
    def from_model(cls, transfered_model, warm_start=False):
        check_is_fitted(transfered_model, ('sigma_square_', 'mu_', 'Sigma_'))
        model = cls()
        model.sigma_square_ = transfered_model.sigma_square_
        model.mu_ = transfered_model.mu_
        model.Sigma_ = transfered_model.Sigma_
        model.alpha_ = transfered_model.alpha_
        model.set_params(**transfered_model.get_params())
        model.set_coef()
        model.features = transfered_model.features
        model.__flag = False
        model.warm_start = warm_start
        return model

    def _fit(self, X, y):
        super().transfer_fit(X, y)


    def _init(self, X, y, warm_start=False):
        # get information of normal equation
        # Trensfer model requires `features` attribute!

        if self.features is None:
            raise Exception('Trensfer model requires `features` attribute.')
        
        super()._init(X, y, warm_start)
        self.__flag = True


# alias for IncrementalBayesianLinearRegression
IncrementalLinearRegression = IncrementalBayesianLinearRegression
TransferLinearRegression = TransferBayesianLinearRegression

if __name__ == '__main__':

    print('receive data')
    X=np.array([[1,2,1],[3,3,2], [4,5,3],[5,6,4]])
    y=np.array([3, 6,9.5,10.5])
    print('create a model (set warm_start=True)')
    ilr = IncrementalLinearRegression(warm_start=True)
    ilr.fit(X, y)
    print(f'''
    coef: {ilr.coef_}
    training score: {ilr.score(X, y)}
    ''')

    print('Saving the model')
    import joblib
    joblib.dump(ilr, 'ilr.model')
    print('Saved the model')
    print('Loading the model')
    ilr = joblib.load('ilr.model')
    print('Receive new data')
    print(f'''
        previous coef: {ilr.coef_}
        ''')
    X = np.array([[5,6,10],[4,3,2], [4,7,6],[5,8,10]])
    y = np.array([11, 8,11,13])
    ilr.fit(X, y)
    print(f'''After incremental learning.
    coef: {ilr.coef_}
    training score: {ilr.score(X, y)}
    informative features: {ilr.informative_features(0.001)}
    ''')

    print('Create a transfer learning model from `ilr`')
    tlr = TransferLinearRegression.from_model(transfered_model=ilr)
    X = np.array([[5,6,10],[4,3,2], [4,7,6],[5,8,10]])
    y = np.array([11, 8,11,13])
    tlr.fit(X, y)
    print(f'''After Transfer learning.
    coef: {tlr.coef_}
    training score: {tlr.score(X, y)}
    informative features: {tlr.informative_features(0.001)}
    ''')
