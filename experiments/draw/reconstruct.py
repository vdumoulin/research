import cPickle
import itertools

import theano
from blocks.graph import ComputationGraph
from fuel.datasets import BinarizedMNIST
from matplotlib import cm, pyplot
from theano import tensor


def reconstruct(model, nrows, ncols):
    dataset = BinarizedMNIST('valid')
    originals, = dataset.get_data(request=range(nrows * ncols))

    figure, axes = pyplot.subplots(nrows=nrows, ncols=ncols)
    for n, (i, j) in enumerate(itertools.product(xrange(nrows),
                                                 xrange(ncols))):
        ax = axes[i][j]
        ax.axis('off')
        ax.imshow(originals[n].reshape((28, 28)), cmap=cm.Greys_r,
                  interpolation='nearest')

    draw = model.top_bricks[0]
    x = tensor.matrix('x')
    x_hat = draw.reconstruct(x)
    computation_graph = ComputationGraph([x_hat])
    f = theano.function([x], x_hat, updates=computation_graph.updates)
    reconstructions = f(originals)

    figure, axes = pyplot.subplots(nrows=nrows, ncols=ncols)
    for n, (i, j) in enumerate(itertools.product(xrange(nrows),
                                                 xrange(ncols))):
        ax = axes[i][j]
        ax.axis('off')
        ax.imshow(reconstructions[n].reshape((28, 28)), cmap=cm.Greys_r,
                  interpolation='nearest')

    pyplot.show()


if __name__ == "__main__":
    nrows = ncols = 10
    with open('vae_model.pkl') as f:
        model = cPickle.load(f)
    reconstruct(model, nrows, ncols)
