from blocks.bricks import Initializable, Linear, Random
from blocks.bricks.base import application
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.graph import Annotation, add_annotation
from theano import tensor


class DRAW(BaseRecurrent, Initializable, Random):
    def __init__(self, nvis, nhid, encoding_mlp, encoding_lstm, decoding_mlp,
                 decoding_lstm, T=1, **kwargs):
        super(DRAW, self).__init__(**kwargs)

        self.nvis = nvis
        self.nhid = nhid
        self.T = T

        self.encoding_mlp = encoding_mlp
        self.encoding_mlp.name = 'encoder_mlp'
        for i, child in enumerate(self.encoding_mlp.children):
            child.name = '{}_{}'.format(self.encoding_mlp.name, i)
        self.encoding_lstm = encoding_lstm
        self.encoding_lstm.name = 'encoder_lstm'
        self.encoding_parameter_mapping = Fork(
            output_names=['mu_phi', 'log_sigma_phi'], prototype=Linear())

        self.h_dec_mapping = Linear(name='h_dec_mapping')

        self.decoding_mlp = decoding_mlp
        self.decoding_mlp.name = 'decoder_mlp'
        for i, child in enumerate(self.decoding_mlp.children):
            child.name = '{}_{}'.format(self.decoding_mlp.name, i)
        self.decoding_lstm = decoding_lstm
        self.decoding_lstm.name = 'decoder_lstm'
        self.decoding_parameter_mapping = Linear(name='mu_theta')

        self.prior_mu = tensor.zeros((self.nhid,))
        self.prior_mu.name = 'prior_mu'
        self.prior_log_sigma = tensor.zeros((self.nhid,))
        self.prior_log_sigma.name = 'prior_log_sigma'

        self.children = [self.encoding_mlp, self.encoding_lstm,
                         self.encoding_parameter_mapping,
                         self.decoding_mlp, self.decoding_lstm,
                         self.decoding_parameter_mapping, self.h_dec_mapping]

    def _push_allocation_config(self):
        # The attention-less read operation concatenates x and x_hat, which
        # is why the input to the encoding MLP is twice the size of x.
        self.encoding_mlp.dims[0] = 2 * self.nvis
        self.encoding_mlp.dims[-1] = 4 * self.encoding_lstm.dim
        self.encoding_parameter_mapping.input_dim = self.encoding_lstm.dim
        self.encoding_parameter_mapping.output_dims = dict(
            mu_phi=self.nhid, log_sigma_phi=self.nhid)
        self.decoding_mlp.dims[0] = self.nhid
        self.decoding_mlp.dims[-1] = 4 * self.decoding_lstm.dim
        self.decoding_parameter_mapping.input_dim = self.decoding_lstm.dim
        self.decoding_parameter_mapping.output_dim = self.nvis
        self.h_dec_mapping.input_dim = self.decoding_lstm.dim
        self.h_dec_mapping.output_dim = 4 * self.encoding_lstm.dim

    def sample(self, num_samples):
        z = self.theano_rng.normal(size=(self.T, num_samples, self.nhid),
                                   avg=self.prior_mu,
                                   std=tensor.exp(self.prior_log_sigma))
        return tensor.nnet.sigmoid(self.decode_z(z)[0][-1])

    @application(inputs=['x'], outputs=['x_hat'])
    def reconstruct(self, x):
        x_sequence = tensor.tile(x.dimshuffle('x', 0, 1), (self.T, 1, 1))
        rval = self.apply(x_sequence)
        return tensor.nnet.sigmoid(rval[0][-1])

    @recurrent(sequences=['z'], contexts=[],
               states=['c_states', 'decoding_states', 'decoding_cells'],
               outputs=['c_states', 'decoding_states', 'decoding_cells'])
    def decode_z(self, z, c_states=None, decoding_states=None,
                 decoding_cells=None):
        h_mlp_theta = self.decoding_mlp.apply(z)
        h_lstm_theta, cells_theta = self.decoding_lstm.apply(
            inputs=h_mlp_theta, states=decoding_states, cells=decoding_cells,
            iterate=False)
        new_c_states = (
            c_states + self.decoding_parameter_mapping.apply(h_lstm_theta))

        return new_c_states, h_lstm_theta, cells_theta

    @recurrent(sequences=['x'], contexts=[],
               states=['c_states', 'encoding_states', 'encoding_cells',
                       'decoding_states', 'decoding_cells'],
               outputs=['c_states', 'encoding_states', 'encoding_cells',
                        'decoding_states', 'decoding_cells', 'mu_phi',
                        'log_sigma_phi'])
    def apply(self, x, c_states=None, encoding_states=None,
              encoding_cells=None, decoding_states=None, decoding_cells=None):
        x_hat = x - tensor.nnet.sigmoid(c_states)
        # Concatenate x and x_hat
        r = tensor.concatenate([x, x_hat], axis=1)
        h_mlp_phi = self.encoding_mlp.apply(r)
        h_lstm_phi, cells_phi = self.encoding_lstm.apply(
            inputs=h_mlp_phi + self.h_dec_mapping.apply(decoding_states),
            states=encoding_states, cells=encoding_cells, iterate=False)
        phi = self.encoding_parameter_mapping.apply(h_lstm_phi)
        mu_phi, log_sigma_phi = phi
        epsilon = self.theano_rng.normal(size=mu_phi.shape, dtype=mu_phi.dtype)
        epsilon.name = 'epsilon'
        z = mu_phi + epsilon * tensor.exp(log_sigma_phi)
        z.name = 'z'
        h_mlp_theta = self.decoding_mlp.apply(z)
        h_lstm_theta, cells_theta = self.decoding_lstm.apply(
            inputs=h_mlp_theta, states=decoding_states, cells=decoding_cells,
            iterate=False)
        new_c_states = (
            c_states + self.decoding_parameter_mapping.apply(h_lstm_theta))

        return (new_c_states, h_lstm_phi, cells_phi, h_lstm_theta, cells_theta,
                mu_phi, log_sigma_phi)

    @application(inputs=['x'], outputs=['log_likelihood_lower_bound'])
    def log_likelihood_lower_bound(self, x):
        x_sequence = tensor.tile(x.dimshuffle('x', 0, 1), (self.T, 1, 1))
        rval = self.apply(x_sequence)
        c_states, mu_phi, log_sigma_phi = rval[0], rval[-2], rval[-1]

        prior_mu = self.prior_mu.dimshuffle('x', 'x', 0)
        prior_log_sigma = self.prior_log_sigma.dimshuffle('x', 'x', 0)
        kl_term = (
            prior_log_sigma - log_sigma_phi +
            0.5 * (
                tensor.exp(2 * log_sigma_phi) + (mu_phi - prior_mu) ** 2
            ) / tensor.exp(2 * prior_log_sigma) - 0.5).sum(axis=2).sum(axis=0)
        kl_term.name = 'kl_term'

        reconstruction_term = - (
            x * tensor.nnet.softplus(-c_states[-1]) +
            (1 - x) * tensor.nnet.softplus(c_states[-1])).sum(axis=1)
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
            return self.encoding_lstm.get_dim('states')
        elif name is 'encoding_cells':
            return self.encoding_lstm.get_dim('cells')
        elif name is 'decoding_states':
            return self.decoding_lstm.get_dim('states')
        elif name is 'decoding_cells':
            return self.decoding_lstm.get_dim('cells')
        else:
            return super(DRAW, self).get_dim(name)
