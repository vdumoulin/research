import logging

import theano
import fuel
from blocks.algorithms import GradientDescent, RMSProp
from blocks.bricks import MLP, Tanh
from blocks.bricks.recurrent import LSTM
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import SerializeMainLoop
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Orthogonal, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import PARAMETER
from fuel.datasets import BinarizedMNIST
from fuel.streams import DataStream, ForceFloatX
from fuel.schemes import SequentialScheme
from theano import tensor

from draw import DRAW

fuel.config.floatX = theano.config.floatX


def main(nvis, nhid, encoding_lstm_dim, decoding_lstm_dim, T=1):
    x = tensor.matrix('features')

    # Construct and initialize model
    encoding_mlp = MLP([Tanh()], [None, None])
    decoding_mlp = MLP([Tanh()], [None, None])
    encoding_lstm = LSTM(dim=encoding_lstm_dim)
    decoding_lstm = LSTM(dim=decoding_lstm_dim)
    draw = DRAW(nvis=nvis, nhid=nhid, T=T, encoding_mlp=encoding_mlp,
                decoding_mlp=decoding_mlp, encoding_lstm=encoding_lstm,
                decoding_lstm=decoding_lstm, biases_init=Constant(0),
                weights_init=Orthogonal())
    draw.push_initialization_config()
    encoding_lstm.weights_init = IsotropicGaussian(std=0.001)
    decoding_lstm.weights_init = IsotropicGaussian(std=0.001)
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
            ProgressBar(),
            Printing()])
    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(nvis=784, nhid=50, T=5, encoding_lstm_dim=200, decoding_lstm_dim=200)
