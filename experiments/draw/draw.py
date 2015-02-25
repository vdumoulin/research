import logging

import numpy
import theano
import fuel
from blocks.algorithms import GradientDescent, RMSProp
from blocks.bricks import Initializable, Linear, Random, MLP, Rectifier, Tanh
from blocks.bricks.base import application
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import BaseRecurrent, recurrent, SimpleRecurrent
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import SerializeMainLoop
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, Annotation, add_annotation
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import PARAMETER
from blocks.utils import shared_floatx
from fuel.datasets import BinarizedMNIST
from fuel.streams import DataStream, ForceFloatX
from fuel.schemes import SequentialScheme
from theano import tensor

fuel.config.floatX = theano.config.floatX


class DRAW(BaseRecurrent, Initializable, Random):
    def __init__(self, nvis, nhid, encoding_mlp, encoding_rnn, decoding_mlp,
                 decoding_rnn, T=1, **kwargs):
        super(DRAW, self).__init__(**kwargs)

        self.nvis = nvis
        self.nhid = nhid
        self.T = T

        self.encoding_mlp = encoding_mlp
        self.encoding_mlp.name = 'encoder_mlp'
        for i, child in enumerate(self.encoding_mlp.children):
            child.name = '{}_{}'.format(self.encoding_mlp.name, i)
        self.encoding_rnn = encoding_rnn
        self.encoding_rnn.name = 'encoder_rnn'
        self.encoding_parameter_mapping = Fork(
            output_names=['mu_phi', 'log_sigma_phi'], prototype=Linear())

        self.decoding_mlp = decoding_mlp
        self.decoding_mlp.name = 'decoder_mlp'
        for i, child in enumerate(self.decoding_mlp.children):
            child.name = '{}_{}'.format(self.decoding_mlp.name, i)
        self.decoding_rnn = decoding_rnn
        self.decoding_rnn.name = 'encoder_rnn'
        self.decoding_parameter_mapping = Linear(name='mu_theta')

        self.prior_mu = tensor.zeros((self.nhid,), name='prior_mu')
        self.prior_log_sigma = tensor.zeros(
            (self.nhid,), name='prior_log_sigma')

        self.children = [self.encoding_mlp, self.encoding_rnn,
                         self.encoding_parameter_mapping,
                         self.decoding_mlp, self.decoding_rnn,
                         self.decoding_parameter_mapping]

    def _push_allocation_config(self):
        # The attention-less read operation concatenates x and x_hat, which
        # is why the input to the encoding MLP is twice the size of x.
        self.encoding_mlp.dims[0] = 2 * self.nvis
        self.encoding_rnn.dim = self.encoding_mlp.dims[-1]
        self.encoding_parameter_mapping.input_dim = self.encoding_rnn.dim
        self.encoding_parameter_mapping.output_dims = dict(
            mu_phi=self.nhid, log_sigma_phi=self.nhid)
        self.decoding_mlp.dims[0] = self.nhid
        self.decoding_rnn.dim = self.decoding_mlp.dims[-1]
        self.decoding_parameter_mapping.input_dim = self.decoding_rnn.dim
        self.decoding_parameter_mapping.output_dim = self.nvis

    @recurrent(sequences=['x'], contexts=[],
               states=['c_states', 'encoding_states', 'decoding_states'],
               outputs=['c_states', 'encoding_states', 'decoding_states',
                        'mu_phi', 'log_sigma_phi'])
    def apply(self, x, c_states=None, encoding_states=None,
              decoding_states=None):
        x_hat = x - tensor.nnet.sigmoid(c_states)
        # Concatenate x and x_hat
        r = tensor.concatenate([x, x_hat], axis=1)
        h_mlp_phi = self.encoding_mlp.apply(r)
        # TODO: add dependency on h_{tm1}^{dec}
        h_rnn_phi = self.encoding_rnn.apply(
            inputs=h_mlp_phi, states=encoding_states, iterate=False)
        phi = self.encoding_parameter_mapping.apply(h_rnn_phi)
        mu_phi, log_sigma_phi = phi
        epsilon = self.theano_rng.normal(size=mu_phi.shape, dtype=mu_phi.dtype)
        epsilon.name = 'epsilon'
        z = mu_phi + epsilon * tensor.exp(log_sigma_phi)
        z.name = 'z'
        h_mlp_theta = self.decoding_mlp.apply(z)
        h_rnn_theta = self.decoding_rnn.apply(
            inputs=h_mlp_theta, states=decoding_states, iterate=False)
        new_c_states = (
            c_states + self.decoding_parameter_mapping.apply(h_rnn_theta))

        return new_c_states, h_rnn_phi, h_rnn_theta, mu_phi, log_sigma_phi

    @application(inputs=['x'], outputs=['log_likelihood_lower_bound'])
    def log_likelihood_lower_bound(self, x):
        x_sequence = tensor.tile(x.dimshuffle('x', 0, 1), (self.T, 1, 1))
        rval = self.apply(x_sequence)
        c_states, mu_phi, log_sigma_phi = rval[0], rval[3], rval[4]

        prior_mu = self.prior_mu.dimshuffle('x', 'x', 0)
        prior_log_sigma = self.prior_log_sigma.dimshuffle('x', 'x', 0)
        kl_term = (
            prior_log_sigma - log_sigma_phi
            + 0.5 * (
                tensor.exp(2 * log_sigma_phi) + (mu_phi - prior_mu) ** 2
            ) / tensor.exp(2 * prior_log_sigma)
            - 0.5).sum(axis=2).mean(axis=0)
        kl_term.name = 'kl_term'

        reconstruction_term = - (
            x * tensor.nnet.softplus(-c_states[-1])
            + (1 - x) * tensor.nnet.softplus(c_states[-1])).sum(axis=1)
        reconstruction_term.name = 'reconstruction_term'

        log_likelihood_lower_bound = reconstruction_term - kl_term
        log_likelihood_lower_bound.name = 'log_likelihood_lower_bound'

        annotation = Annotation()
        annotation.add_auxiliary_variable(kl_term, name='kl_term')
        annotation.add_auxiliary_variable(-reconstruction_term,
                                          name='reconstruction_term')
        add_annotation(log_likelihood_lower_bound, annotation)

        return log_likelihood_lower_bound

    def get_dim(self, name):
        if name is 'c_states':
            return self.nvis
        elif name is 'encoding_states':
            return self.encoding_rnn.get_dim('states')
        elif name is 'decoding_states':
            return self.decoding_rnn.get_dim('states')
        else:
            return super(DRAW, self).get_dim(name)


def main(nvis, nhid, nlat, T=1):
    x = tensor.matrix('features')

    # Construct and initialize model
    encoding_mlp = MLP([Rectifier()], [None, nhid])
    decoding_mlp = MLP([Rectifier()], [None, nhid])
    encoding_rnn = SimpleRecurrent(activation=Tanh())
    decoding_rnn = SimpleRecurrent(activation=Tanh())
    draw = DRAW(nvis=nvis, nhid=nlat, T=T, encoding_mlp=encoding_mlp,
                decoding_mlp=decoding_mlp, encoding_rnn=encoding_rnn,
                decoding_rnn=decoding_rnn, biases_init=Constant(0),
                weights_init=IsotropicGaussian(std=0.001))
    draw.initialize()

    # Compute cost
    cost = -draw.log_likelihood_lower_bound(x).mean()
    cost.name = 'nll_upper_bound'
    model = Model(cost)

    # Datasets and data streams
    mnist_train = BinarizedMNIST('train')
    train_loop_stream = ForceFloatX(DataStream(
        dataset=mnist_train,
        iteration_scheme=SequentialScheme(mnist_train.num_examples, 100)))
    train_monitor_stream = ForceFloatX(DataStream(
        dataset=mnist_train,
        iteration_scheme=SequentialScheme(mnist_train.num_examples, 500)))
    mnist_valid = BinarizedMNIST('valid')
    valid_monitor_stream = ForceFloatX(DataStream(
        dataset=mnist_valid,
        iteration_scheme=SequentialScheme(mnist_valid.num_examples, 500)))
    mnist_test = BinarizedMNIST('test')
    test_monitor_stream = ForceFloatX(DataStream(
        dataset=mnist_test,
        iteration_scheme=SequentialScheme(mnist_test.num_examples, 500)))

    # Get parameters and monitoring channels
    computation_graph = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(computation_graph.variables)
    monitoring_channels = dict([
        ('avg_' + channel.tag.name, channel.mean()) for channel in
        VariableFilter(name='.*term$')(computation_graph.auxiliary_variables)])
    for name, channel in monitoring_channels.items():
        channel.name = name
    monitored_quantities = monitoring_channels.values() + [cost]

    # Training loop
    step_rule = RMSProp(learning_rate=1e-3, decay_rate=0.95)
    algorithm = GradientDescent(cost=cost, params=params, step_rule=step_rule)
    algorithm.add_updates(computation_graph.updates)
    main_loop = MainLoop(
        model=model, data_stream=train_loop_stream, algorithm=algorithm,
        extensions=[
            Timing(),
            SerializeMainLoop('vae.pkl', save_separately=['model']),
            FinishAfter(after_n_epochs=200),
            DataStreamMonitoring(
                monitored_quantities, train_monitor_stream, prefix="train",
                updates=computation_graph.updates),
            DataStreamMonitoring(
                monitored_quantities, valid_monitor_stream, prefix="valid",
                updates=computation_graph.updates),
            DataStreamMonitoring(
                monitored_quantities, test_monitor_stream, prefix="test",
                updates=computation_graph.updates),
            Printing()])
    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(nvis=784, nhid=500, nlat=100, T=2)
