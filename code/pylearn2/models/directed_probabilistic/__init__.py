"""
Directed probabilistic models
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"


import numpy
import theano.tensor as T
from theano.compat import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.models.model import Model
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.space import VectorSpace, NullSpace


theano_rng = RandomStreams(seed=23541)


class Distribution(Model):
    """
    WRITEME
    """
    def _initialize_weights(self, dim_0, dim_1):
        """
        Initialize a (dim_0, dim_1)-shaped weight matrix

        Parameters
        ----------
        dim_0 : int
            First dimension of the weights matrix
        dim_1 : int
            Second dimension of the weights matrix

        Returns
        -------
        rval : `numpy.ndarray`
            A (dim_0, dim_1)-shaped, properly initialized weights matrix
        """
        rval = (2 * numpy.random.normal(size=(dim_0, dim_1)) - 1) / dim_0
        return rval

    def get_layer_monitoring_channels(self):
        rval = OrderedDict()

        for param in self.get_params():
            rval[param.name + "_min"] = param.min()
            rval[param.name + "_max"] = param.max()
            rval[param.name + "_mean"] = param.mean()

        return rval


class JointDistribution(Distribution):
    def _sample(self, num_samples):
        raise NotImplementedError()

    def sample(self, num_samples, return_log_likelihood=False,
               return_probabilities=False):
        """
        Samples from the modeled joint distribution p(x)

        Parameters
        ----------
        num_samples : int
            Number of samples to draw
        return_log_likelihood : bool, optional
            If `True`, returns the log-likelihood of the samples in addition to
            the samples themselves. Defaults to `False`.
        return_probabilities : bool, optional
            If `True`, returns the probabilities from which samples were drawn
            in addition to the samples themselves. Defaults to `False`.

        Returns
        -------
        samples : tensor-like
            Batch of `num_samples` samples from p(x)
        log_likelihood : tensor-like, optional
            Log-likelihood of the drawn samples according to p(x). Returned
            only if `return_log_likelihood` is set to `True`.
        probabilities : tensor-like, optional
            Probabilities from which the samples were drawn. Returned only if
            `return_probabilities` is set to `True`.
        """
        rval = self._sample(num_samples=num_samples)
        samples, log_likelihood, probabilities = rval

        if not return_log_likelihood and not return_probabilities:
            return samples
        else:
            rval = [samples]
            if return_log_likelihood:
                rval.append(log_likelihood)
            if return_probabilities:
                rval.append(probabilities)
            return tuple(rval)

    def _log_likelihood(self, X):
        raise NotImplementedError()

    def log_likelihood(self, X):
        """
        Computes the log-likelihood of a batch of observed examples on a
        per-example basis

        Parameters
        ----------
        X : tensor-like
            Batch of observed examples

        Returns
        -------
        rval : tensor-like
            Log-likelihood for the batch of visible examples, with shape
            (X.shape[0],)
        """
        return self._log_likelihood(X=X)


class ConditionalDistribution(Distribution):
    def _sample(self, num_samples):
        raise NotImplementedError()

    def sample(self, Y, return_log_likelihood=False,
               return_probabilities=False):
        """
        Samples from the conditional distribution p(x | y)

        Parameters
        ----------
        return_log_likelihood : bool, optional
            If `True`, returns the conditional log-likelihood of the samples in
            addition to the samples themselves. Defaults to `False`.
        return_probabilities : bool, optional
            If `True`, returns the conditional probabilities from which samples
            were drawn in addition to the samples themselves. Defaults to
            `False`.

        Returns
        -------
        samples : tensor-like
            Batch of `num_samples` samples from p(x)
        log_likelihood : tensor-like, optional
            Log-likelihood of the drawn samples according to p(x | y). Returned
            only if `return_log_likelihood` is set to `True`.
        probabilities : tensor-like, optional
            Probabilities from which the samples were drawn. Returned only if
            `return_probabilities` is set to `True`.
        """
        rval = self._sample(Y=Y)
        samples, log_likelihood, probabilities = rval

        if not return_log_likelihood and not return_probabilities:
            return samples
        else:
            rval = [samples]
            if return_log_likelihood:
                rval.append(log_likelihood)
            if return_probabilities:
                rval.append(probabilities)
            return tuple(rval)

    def _log_likelihood(self, X, Y):
        raise NotImplementedError()

    def log_likelihood(self, X, Y):
        """
        Computes the conditional log-likelihood of a batch of observed examples
        on a per-example basis

        Parameters
        ----------
        X : tensor-like
            Batch of observed examples
        Y : tensor-like
            Batch of conditioning examples

        Returns
        -------
        rval : tensor-like
            Conditional Log-likelihood for the batch of visible examples, with
            shape (X.shape[0],)
        """
        return self._log_likelihood(X=X, Y=Y)


class ProductOfBernoulli(JointDistribution):
    """
    Random binary vector whose distribution is a product of Bernoulli
    distributions, i.e.

        p(v) = \prod_i v_i ** p_i * (1 - v_i) ** (1 - p_i)
    """
    def __init__(self, dim):
        """
        Parameters
        ----------
        dim : int
            Dimension of the random binary vector
        """
        self.dim = dim

        # Parameter initialization
        b_value = numpy.zeros(self.dim)
        self.b = sharedX(b_value, 'b')
        self.p = T.nnet.sigmoid(self.b)

        # Space initialization
        self.input_space = NullSpace()
        self.output_space = VectorSpace(dim=self.dim)

    def _sample(self, num_samples):
        samples = theano_rng.uniform((num_samples, self.dim)) <= self.p
        log_likelihood = self.log_likelihood(samples)
        probabilities = T.zeros_like(samples) + self.p
        return samples, log_likelihood, probabilities

    def _log_likelihood(self, X):
        return (X * T.log(self.p) + (1 - X) * T.log(1 - self.p)).sum(axis=1)

    @wraps(Model.get_params)
    def get_params(self):
        return [self.b]


class StochasticSigmoid(ConditionalDistribution):
    """
    Implements the conditional distribution of a random binary vector x given
    an input vector y as a product of Bernoulli distributions, i.e.

        p(x | y) = \prod_i p(x_i | y),

    where

        p(x_i | y) = sigmoid(y.W_i + b_i)
    """
    def __init__(self, dim, dim_cond, clamp_sigmoid=False):
        """
        Parameters
        ----------
        dim : int
            Dimension of the modeled vector x
        dim_cond : int
            Dimension of the conditioning vector y
        """
        self.dim_cond = dim_cond
        self.dim = dim
        self.clamp_sigmoid = clamp_sigmoid

        # Bias initialization
        b_value = numpy.zeros(self.dim)
        self.b = sharedX(b_value, 'b')

        # Weights initialization
        W_value = self._initialize_weights(self.dim_cond, self.dim)
        self.W = sharedX(W_value, 'W')

        # Space initialization
        self.input_space = VectorSpace(dim=self.dim_cond)
        self.target_space = VectorSpace(dim=self.dim)

    def _sigmoid(self, x):
        """
        WRITEME

        Parameters
        ----------
        x : WRITEME
        """
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    def _sample(self, Y):
        batch_size = Y.shape[0]
        probabilities = self._sigmoid(T.dot(Y, self.W) + self.b)
        samples = theano_rng.uniform((batch_size, self.dim)) <= probabilities
        log_likelihood = (
            samples * T.log(probabilities) +
            (1 - samples) * T.log(1 - probabilities)
        ).sum(axis=1)
        
        return samples, log_likelihood, probabilities

    def _log_likelihood(self, X, Y):
        p = self._sigmoid(T.dot(Y, self.W) + self.b)
        return (X * T.log(p) + (1 - X) * T.log(1 - p)).sum(axis=1)

    @wraps(Model.get_params)
    def get_params(self):
        return [self.W, self.b]

    def get_weights(self):
        return self.W.get_value()
