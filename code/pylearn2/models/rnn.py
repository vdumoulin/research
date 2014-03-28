"""
WRITEME
"""
import numpy
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace
from research.code.pylearn2.space import VectorSequenceSpace
from research.code.pylearn2.datasets.timit import TIMITSequences
from pylearn2.utils import sharedX
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class ToyRNN(Model):
    """
    WRITEME
    """
    def __init__(self, nvis, nhid, irange=0.05, non_linearity='sigmoid',
                 use_ground_truth=True):
        allowed_non_linearities = {'sigmoid': T.nnet.sigmoid,
                                   'tanh': T.tanh}
        self.nvis = nvis
        self.nhid = nhid
        self.use_ground_truth = use_ground_truth
        self.alpha = sharedX(1)
        self.alpha_decrease_rate = 0.9

        assert non_linearity in allowed_non_linearities
        self.non_linearity = allowed_non_linearities[non_linearity]

        # Space initialization
        self.input_space = CompositeSpace([
            VectorSequenceSpace(dim=self.nvis),
            VectorSequenceSpace(dim=62)
        ])
        self.output_space = VectorSequenceSpace(dim=1)
        self.input_source = ('features', 'phones')
        self.target_source = 'targets'

        # Features-to-hidden matrix
        W_value = numpy.random.uniform(low=-irange, high=irange,
                                       size=(self.nvis, self.nhid))
        self.W = sharedX(W_value, name='W')
        # Phones-to-hidden matrix
        V_value = numpy.random.uniform(low=-irange, high=irange,
                                       size=(62, self.nhid))
        self.V = sharedX(V_value, name='V')
        # Hidden-to-hidden matrix
        M_value = numpy.random.uniform(low=-irange, high=irange,
                                       size=(self.nhid, self.nhid))
        self.M = sharedX(M_value, name='M')
        # Hidden biases
        b_value = numpy.zeros(self.nhid)
        self.b = sharedX(b_value, name='b')
        # Hidden-to-out matrix
        U_value = numpy.random.uniform(low=-irange, high=irange,
                                       size=(self.nhid, 1))
        self.U = sharedX(U_value, name='U')
        # Output bias
        c_value = numpy.zeros(1)
        self.c = sharedX(c_value, name='c')

    def fprop_step(self, features, phones, h_tm1, out):
        h = self.non_linearity(T.dot(features, self.W) +
                               T.dot(phones, self.V) +
                               T.dot(h_tm1, self.M) +
                               self.b)
        out = T.dot(h, self.U) + self.c
        return h, out

    def fprop_step_prime(self, truth, phones, features, h_tm1, out):
        T.set_subtensor(features[-1],
                        (1 - self.alpha) * features[-1] + self.alpha * truth[-1])
        h = self.non_linearity(T.dot(features, self.W) +
                               T.dot(phones, self.V) +
                               T.dot(h_tm1, self.M) +
                               self.b)
        out = T.dot(h, self.U) + self.c
        features = T.concatenate([features[1:], out])
        return features, h, out

    def fprop(self, data):
        if self.use_ground_truth:
            self.input_space.validate(data)
            features, phones = data

            init_h = T.alloc(numpy.cast[theano.config.floatX](0), self.nhid)
            init_out = T.alloc(numpy.cast[theano.config.floatX](0), 1)
            init_out = T.unbroadcast(init_out, 0)

            fn = lambda f, p, h, o: self.fprop_step(f, p, h, o)

            ((h, out), updates) = theano.scan(fn=fn,
                                              sequences=[features, phones],
                                              outputs_info=[dict(initial=init_h,
                                                                 taps=[-1]),
                                                            init_out])
            return out
        else:
            self.input_space.validate(data)
            features, phones = data

            init_in = features[0]
            init_h = T.alloc(numpy.cast[theano.config.floatX](0), self.nhid)
            init_out = T.alloc(numpy.cast[theano.config.floatX](0), 1)
            init_out = T.unbroadcast(init_out, 0)

            fn = lambda t, p, f, h, o: self.fprop_step_prime(t, p, f, h, o)

            ((f, h, out), updates) = theano.scan(fn=fn,
                                                 sequences=[features, phones],
                                                 outputs_info=[init_in,
                                                               dict(initial=init_h,
                                                                    taps=[-1]),
                                                               init_out])
            return out

    def predict_next(self, features, phones, h_tm1):
        h = T.nnet.sigmoid(T.dot(features, self.W) +
                           T.dot(phones, self.V) +
                           T.dot(h_tm1, self.M) +
                           self.b)
        out = T.dot(h, self.U) + self.c
        return h, out

    def get_params(self):
        return [self.W, self.V, self.M, self.b, self.U, self.c]

    def get_input_source(self):
        return self.input_source

    def get_target_source(self):
        return self.target_source

    def censor_updates(self, updates):
        updates[self.alpha] = self.alpha_decrease_rate * self.alpha

    def get_monitoring_channels(self, data):
        rval = OrderedDict()
        rval['alpha'] = self.alpha
        return rval


class RNNCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        """
        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        predictions = model.fprop(inputs)
        return T.mean(T.sqr(targets - predictions))


if __name__ == "__main__":
    model = ToyRNN(nvis=100, nhid=100)
    cost = RNNCost()

    features = T.matrix('features')
    phones = T.matrix('phones')
    targets = T.matrix('targets')

    cost_expression = cost.expr(model, ((features, phones), targets))

    fn = theano.function(inputs=[features, phones, targets],
                         outputs=cost_expression)

    valid_timit = TIMITSequences("valid", frame_length=100, audio_only=False)
    data_specs = (CompositeSpace([VectorSequenceSpace(dim=100),
                                  VectorSequenceSpace(dim=1),
                                  VectorSequenceSpace(dim=62)]),
                  ('features', 'targets', 'phones'))
    it = valid_timit.iterator(mode='sequential', data_specs=data_specs,
                              num_batches=10, batch_size=1)
    for f, t, p in it:
        print f.shape, p.shape, t.shape
        print fn(f, p, t)
