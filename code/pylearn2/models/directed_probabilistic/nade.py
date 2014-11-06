"""
Neural autoregressive density estimator (NADE) implementation
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Jorg Bornschein", "Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"


import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.models.model import Model
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace
# from research.code.pylearn2.utils.unrolled_scan import unrolled_scan
from research.code.pylearn2.models.directed_probabilistic import (
    JointDistribution, ConditionalDistribution
)


theano_rng = RandomStreams(seed=2341)


class NADEBase(Model):
    """
    WRITEME
    """
    def __init__(self, dim, dim_hid, clamp_sigmoid=False, unroll_scan=1):
        """
        Parameters
        ----------
        dim : int
            Number of observed binary variables
        dim_hid : int
            Number of latent binary variables
        clamp_sigmoid : bool, optional
            WRITEME. Defaults to `False`.
        unroll_scan : int, optional
            WRITEME. Defaults to 1.
        """
        super(NADEBase, self).__init__()

        self.dim = dim
        self.dim_hid = dim_hid
        self.clamp_sigmoid = clamp_sigmoid
        self.unroll_scan = unroll_scan

        self.input_space = VectorSpace(dim=self.dim)

        # Visible biases
        b_value = numpy.zeros(self.dim)
        self.b = sharedX(b_value, 'b')
        # Hidden biases
        c_value = numpy.zeros(self.dim_hid)
        self.c = sharedX(c_value, 'c')
        # Encoder weights
        W_value = self._initialize_weights(self.dim, self.dim_hid)
        self.W = sharedX(W_value, 'W')
        # Decoder weights
        V_value = self._initialize_weights(self.dim_hid, self.dim)
        self.V = sharedX(V_value, 'V')

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

    def sigmoid(self, x):
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

    def get_params(self):
        """
        Returns
        -------
        params : list of tensor-like
            The model's parameters
        """
        return [self.b, self.c, self.W, self.V]

    def get_weights(self):
        """
        Aliases to `NADE.get_encoder_weights`
        """
        return self.get_encoder_weights()

    def set_weights(self, weights):
        """
        Aliases to `NADE.set_encoder_weights`
        """
        self.set_encoder_weights(weights)

    def get_encoder_weights(self):
        """
        Returns
        -------
        rval : `numpy.ndarray`
            Encoder weights
        """
        return self.W.get_value()

    def set_encoder_weights(self, weights):
        """
        Sets encoder weight values

        Parameters
        ----------
        weights : `numpy.ndarray`
            Encoder weight values to assign to self.W
        """
        self.W.set_value(weights)

    def get_decoder_weights(self):
        """
        Returns
        -------
        rval : `numpy.ndarray`
            Decoder weights
        """
        return self.V.get_value()

    def set_decoder_weights(self, weights):
        """
        Sets decoder weight values

        Parameters
        ----------
        weights : `numpy.ndarray`
            Decoder weight values to assign to self.V
        """
        self.V.set_value(weights)

    def get_visible_biases(self):
        """
        Returns
        -------
        rval : `numpy.ndarray`
            Visible biases
        """
        return self.b.get_value()

    def set_visible_biases(self, biases):
        """
        Sets visible bias values

        Parameters
        ----------
        biases : `numpy.ndarray`
            Visible bias values to assign to self.b
        """
        self.b.set_value(biases)

    def get_hidden_biases(self):
        """
        Returns
        -------
        rval : `numpy.ndarray`
            Hidden biases
        """
        return self.c.get_value()

    def set_hidden_biases(self, biases):
        """
        Sets hidden bias values

        Parameters
        ----------
        biases : `numpy.ndarray`
            Hidden bias values to assign to self.c
        """
        self.c.set_value(biases)

    def _base_log_likelihood(self, X, W, V, b, c):
        """
        Computes the log-likelihood of a batch of visible examples

        Parameters
        ----------
        X : tensor-like
            Batch of visible examples
        W : tensor-like
            Encoder weights
        V : tensor-like
            Decoder weights
        b : tensor-like
            Visible biases
        c : tensor-like
            Hidden biases

        Returns
        -------
        rval : tensor-like
            Log-likelihood for the batch of visible examples
        """
        # Transformation matrix. A 3D tensor of the form
        #
        # [[[      0,       0, ...,       0],
        #   [X[0, 0],       0, ...,       0],
        #   [X[0, 0], X[0, 1], ...,       0],
        #   [X[0, 0], X[0, 1], ..., X[0, d]]],
        #                ...
        #  [[      0,       0, ...,       0],
        #   [X[n, 0],       0, ...,       0],
        #   [X[n, 0], X[n, 1], ...,       0],
        #   [X[n, 0], X[n, 1], ..., X[n, d]]]]
        #
        # Its purpose is make the `W_{., <i} v_{<i}` matrix product for all
        # examples in the X batch.
        M = (X.dimshuffle(0, 'x', 1) * T.ones((X.shape[1], X.shape[1])) *
             numpy.asarray(numpy.tril(numpy.ones((self.dim, self.dim)),
                                      k=-1),
                           dtype=theano.config.floatX))
        # The dot product of M with W will produce a tensor of the form
        #
        # [[[ h_1(X[0, :]) ],
        #       ...
        #   [ h_d(X[0, :]) ]],
        #       ...
        #  [[ h_1(X[n, :]) ],
        #       ...
        #   [ h_d(X[n, :]) ]]]
        #
        h = self.sigmoid(T.dot(M, W) + c)
        # The elementwise product of V.T and h (where V.T is broadcasted to a
        # 3D tensor with one V.T for each example in the X batch) and sum over
        # the h_i axis will produce a matrix whose rows correspond to examples
        # in the X batch and colomns are (W.T)_i h_i.
        p = self.sigmoid((V.T * h).sum(axis=2) + b)

        return (X * T.log(p) + (1 - X) * T.log(1 - p)).sum(axis=1)

    def _base_scan_log_likelihood(self, X, W, V, b, c):
        """
        A slower, Scan version of `NADE._log_likelihood`.

        Parameters
        ----------
        X : tensor-like
            Batch of visible examples
        W : tensor-like
            Encoder weights
        V : tensor-like
            Decoder weights
        b : tensor-like
            Visible biases
        c : tensor-like
            Hidden biases

        Returns
        -------
        rval : tensor-like
            Log-likelihood for the batch of visible examples
        """
        batch_size = X.shape[0]
        # Accumulator for hidden layer activations, initialized with bias
        # values broadcasted to X's shape
        a_init = T.zeros([batch_size, self.dim_hid],
                         dtype=theano.config.floatX) + c
        # Accumulator for log-posterior distribution log p(v), initialized with
        # zeroes
        log_p_init = T.zeros([batch_size], dtype=theano.config.floatX)

        # Function computing -log p(v_i)
        def one_iter(v_i, W_i, V_i, b_i, a, log_p):
            h_i = self.sigmoid(a)
            p_i = self.sigmoid(T.dot(h_i, V_i) + b_i)
            log_p += v_i * T.log(p_i) + (1 - v_i) * T.log(1 - p_i)
            a += T.outer(v_i, W_i)
            return a, log_p

        [a, log_p], updates = unrolled_scan(fn=one_iter,
                                            sequences=[X.T, W, V.T, b],
                                            outputs_info=[a_init, log_p_init],
                                            unroll=self.unroll_scan)
        rval = log_p[-1, :]
        return rval

    def _base_sample(self, num_samples, W, V, b, c):
        """
        Samples from p(v)

        Parameters
        ----------
        num_samples : int
            Number of samples to draw
        W : tensor-like
            Encoder weights
        V : tensor-like
            Decoder weights
        b : tensor-like
            Visible biases
        c : tensor-like
            Hidden biases

        Returns
        -------
        v : tensor-like
            Batch of `num_samples` samples from p(v)
        p : tensor-like
            Joint probability distribution from which v was drawn
        log_likelihood : tensor-like
            Log-likelihood of the batch of samples
        """
        # Accumulator for hidden layer activations, initialized with bias
        # values and broadcasted to the right shape
        a_init = T.zeros((num_samples, self.dim_hid),
                         dtype=theano.config.floatX) + c
        # Accumulator for conditional distribution p(v_i | v_{<i}), initialized
        # with zeroes
        p_init = T.zeros((num_samples,), dtype=theano.config.floatX)
        # Accumulator for visible samples, initialized with zeroes
        v_init = T.zeros((num_samples,), dtype=theano.config.floatX)
        # Accumulator for log-likelihood log p(v) of visible samples
        log_likelihood_init = T.zeros((num_samples,),
                                      dtype=theano.config.floatX)

        # Function sampling from p(v_i)
        def one_iter(W_i, V_i, b_i, a, v_lt_i, p_lt_i, log_likelihood):
            h_i = self.sigmoid(a)
            p_i = self.sigmoid(T.dot(h_i, V_i) + b_i)
            v_i = 1. * (theano_rng.uniform([num_samples]) <= p_i)
            log_likelihood += v_i * T.log(p_i) + (1 - v_i) * T.log(1 - p_i)
            a += T.outer(v_i, W_i)
            return a, v_i, p_i, log_likelihood

        [a, v, p, log_likelihood], updates = unrolled_scan(
            fn=one_iter,
            sequences=[W, V.T, b],
            outputs_info=[a_init, v_init, p_init, log_likelihood_init],
            unroll=self.unroll_scan
        )

        return v.T, log_likelihood[-1, :], p.T


class NADE(NADEBase, JointDistribution):
    """
    An implementation of Larochelle's and Murray's neural autoregressive
    density estimator (NADE)
    """
    def _log_likelihood(self, X):
        return self._base_log_likelihood(X, self.W, self.V, self.b, self.c)

    def _sample(self, num_samples):
        return self._base_sample(num_samples, self.W, self.V, self.b, self.c)


class CNADE(NADEBase, ConditionalDistribution):
    """
    An implementation of Larochelle's and Murray's neural autoregressive
    density estimator (NADE) conditioned on an external observation
    """
    def __init__(self, dim, dim_hid, dim_cond, clamp_sigmoid=False, unroll_scan=1):
        """
        Parameters
        ----------
        dim : int
            Number of observed binary variables
        dim_hid : int
            Number of latent binary variables
        dim_cond : int
            Number of conditioning variables
        clamp_sigmoid : bool, optional
            WRITEME. Defaults to `False`.
        unroll_scan : int, optional
            WRITEME. Defaults to 1.
        """
        super(CNADE, self).__init__(dim=dim, dim_hid=dim_hid,
                                    clamp_sigmoid=clamp_sigmoid,
                                    unroll_scan=unroll_scan)

        self.dim_cond = dim_cond

        # Conditioning weights matrix for visible biases
        U_b_value = self._initialize_weights(self.dim_cond, self.dim)
        self.U_b = sharedX(U_b_value, 'U_b')
        # Conditioning weights matrix for hidden biases
        U_c_value = self._initialize_weights(self.dim_cond, self.dim_hid)
        self.U_c = sharedX(U_c_value, 'U_c')

    def get_params(self):
        """
        Returns
        -------
        params : list of tensor-like
            The model's parameters
        """
        params = super(CNADE, self).get_params()
        params.extend([self.U_c, self.U_b])
        return params

    def _log_likelihood(self, X, Y):
        # Conditioned visible biases, shape is (batch_size, self.dim_hid)
        b_cond = self.b + T.dot(Y, self.U_b)
        # Conditioned hidden biases, shape is (batch_size, self.dim_hid)
        c_cond = self.c + T.dot(Y, self.U_c)

        return self._base_log_likelihood(X, self.W, self.V, b_cond, c_cond)

    def _sample(self, Y):
        num_samples = Y.shape[0]
        # Conditioned visible biases, shape is (batch_size, self.dim_hid)
        b_cond = self.b + T.dot(Y, self.U_b)
        # Conditioned hidden biases, shape is (batch_size, self.dim_hid)
        c_cond = self.c + T.dot(Y, self.U_c)

        # Here we give b_cond.T as argument because it will be used in a scan
        # loop which systematically loops over the first axis.
        return self._base_sample(num_samples, self.W, self.V, b_cond.T, c_cond)

    def get_visible_conditioning_weights(self):
        """
        Returns
        -------
        rval : `numpy.ndarray`
            Visible conditioning weights
        """
        return self.U_b.get_value()

    def set_visible_conditioning_weights(self, weights):
        """
        Sets visible conditioning weight values

        Parameters
        ----------
        weights : `numpy.ndarray`
            Visible conditioning weight values to assign to self.U_b
        """
        self.U_b.set_value(weights)

    def get_hidden_conditioning_weights(self):
        """
        Returns
        -------
        rval : `numpy.ndarray`
            Hidden conditioning weights
        """
        return self.U_c.get_value()

    def set_hidden_conditioning_weights(self, weights):
        """
        Sets hidden conditioning weight values

        Parameters
        ----------
        weights : `numpy.ndarray`
            Hidden conditioning weight values to assign to self.U_c
        """
        self.U_c.set_value(weights)
