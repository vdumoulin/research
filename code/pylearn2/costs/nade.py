"""
Neural autoregressive density estimator (NADE)-related costs
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"


import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.utils import wraps


class NADECost(DefaultDataSpecsMixin, Cost):
    """
    NADE negative log-likelihood
    """
    @wraps(Cost.expr)
    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X = data
        return -T.mean(model.log_likelihood(X))


class CNADECost(DefaultDataSpecsMixin, Cost):
    """
    CNADE negative log-likelihood
    """
    supervised = True

    @wraps(Cost.expr)
    def expr(self, model, data, ** kwargs):
        self.get_data_specs(model)[0].validate(data)
        X, Y = data
        return -T.mean(model.log_likelihood(X, Y))
