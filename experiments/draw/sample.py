import cPickle
import theano
import itertools
from matplotlib import cm, pyplot


def sample_from_model(model, nrows, ncols):
    draw = model.top_bricks[0]
    f = theano.function([], draw.sample(nrows * ncols))
    samples = f()
    figure, axes = pyplot.subplots(nrows=nrows, ncols=ncols)
    for n, (i, j) in enumerate(itertools.product(xrange(nrows),
                                                 xrange(ncols))):
        ax = axes[i][j]
        ax.axis('off')
        ax.imshow(samples[n].reshape((28, 28)), cmap=cm.Greys_r,
                  interpolation='nearest')
    pyplot.show()


if __name__ == "__main__":
    nrows = ncols = 10
    with open('vae_model.pkl') as f:
        model = cPickle.load(f)
    sample_from_model(model, nrows, ncols)
