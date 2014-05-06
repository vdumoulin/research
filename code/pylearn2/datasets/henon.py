"""
Class for creating Henon map datasets.
"""
__authors__ = ["Dustin Webb"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

import functools
import numpy
import theano
from pylearn2.datasets import Dataset
from pylearn2.space import CompositeSpace, VectorSpace, VectorSequenceSpace
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils import safe_zip
from research.code.pylearn2.utils.iteration import FiniteDatasetIterator


class HenonMapSequences(Dataset):
    """
    Generates data for Henon map, i.e.

       x_{n+1} = 1 - \alpha*x_n^2 + y_n
       y_{n+1} = \beta*x_n

    Parameters
    ----------
    alpha_list : list of float, optional
        WRITEME
    beta_list : list of float, optional
        WRITEME
    init_state_list : list of ndarray, optional
        WRITEME
    num_samples : int, optional
        Number of desired samples per sequence. Defaults to 1000.
    frame_length : int
        Number of samples contained in a frame. Must divide samples.
    rng : int, optional
        Seed for random number generator. Defaults to None, in which case the
        default seed will be used.
    """
    _default_seed = 1

    def __init__(self, alpha_list=[1.4], beta_list=[0.3],
                 init_state_list=[numpy.array([0, 0])], num_samples=1000,
                 rng=None):
        # Validate parameters and set member variables
        self.alpha_list = alpha_list
        self.beta_list = beta_list

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.num_samples = num_samples
        self.num_examples = len(alpha_list)

        self.init_state_list = init_state_list

        # Initialize RNG
        if rng is None:
            self.rng = numpy.random.RandomState(self._default_seed)
        else:
            self.rng = numpy.random.RandomState(rng)

        X, y = self._generate_data()
        self.data = (X, y)

        # DataSpecs
        features_space = VectorSequenceSpace(dim=2)
        features_source = 'features'

        targets_space = VectorSequenceSpace(dim=2)
        targets_source = 'targets'

        space = CompositeSpace([features_space, targets_space])
        source = tuple([features_source, targets_source])
        self.data_specs = (space, source)

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('shuffled_sequential')
        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))

    def _validate_source(self, source):
        """
        Verify that all sources in the source tuple are provided by the
        dataset. Raise an error if some requested source is not available.

        Parameters
        ----------
        source : `tuple` of `str`
            Requested sources
        """
        for s in source:
            try:
                self.data_specs[1].index(s)
            except ValueError:
                raise ValueError("the requested source named '" + s + "' " +
                                 "is not provided by the dataset")

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `TIMIT.get()`
            _does_ respect those data specs.
        """
        return self.data_specs

    def get(self, source, indexes):
        """
        .. todo::

            WRITEME
        """
        if type(indexes) is slice:
            indexes = numpy.arange(indexes.start, indexes.stop)
        assert indexes.shape == (1,)
        self._validate_source(source)
        rval = []
        for so in source:
            rval.append(
                self.data[self.data_specs[1].index(so)][indexes]
            )
        return tuple(rval)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        """
        .. todo::

            WRITEME
        """
        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            convert.append(None)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.num_examples, batch_size,
                                          num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

    def _generate_data(self):
        """
        Generates X matrix for DenseDesignMatrix initialization
        function.
        """
        X = []
        y = []

        for init_state, alpha, beta in zip(self.init_state_list,
                                           self.alpha_list,
                                           self.beta_list):
            rval = self._generate_sequence(self.num_samples, init_state, alpha,
                                           beta)
            X.append(rval[0])
            y.append(rval[0])

        return X, y

    def _generate_sequence(self, num_samples, init_state, alpha, beta):
        # Initialize arrays
        X = numpy.zeros((num_samples + 1, 2), dtype=theano.config.floatX)
        X[0, :] = init_state

        # Generate sequence
        for i in range(1, X.shape[0]):
            X[i, 0] = 1 - alpha * X[i - 1, 0] ** 2 + X[i - 1, 1]
            X[i, 1] = beta * X[i - 1, 0]

        return X[:-1, :], X[1:, :]


class HenonMap(Dataset):
    """
    Generates data for Henon map, i.e.

       x_{n+1} = 1 - \alpha*x_n^2 + y_n
       y_{n+1} = \beta*x_n

    Parameters
    ----------
    alpha_list : list of float, optional
        WRITEME
    beta_list : list of float, optional
        WRITEME
    init_state_list : list of ndarray, optional
        WRITEME
    num_samples : int, optional
        Number of desired samples per sequence. Defaults to 1000.
    frame_length : int
        Number of samples contained in a frame. Must divide samples.
    rng : int, optional
        Seed for random number generator. Defaults to None, in which case the
        default seed will be used.
    """
    _default_seed = 1

    def __init__(self, alpha_list=[1.4], beta_list=[0.3],
                 init_state_list=[numpy.array([0, 0])], num_samples=1000,
                 frame_length=1, rng=None):
        # Validate parameters and set member variables
        self.alpha_list = alpha_list
        self.beta_list = beta_list

        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.num_samples = num_samples
        self.num_examples = len(alpha_list)
        self.frame_length = frame_length

        self.init_state_list = init_state_list

        # Initialize RNG
        if rng is None:
            self.rng = numpy.random.RandomState(self._default_seed)
        else:
            self.rng = numpy.random.RandomState(rng)

        X, y = self._generate_data()
        self.data = (X, y)

        # DataSpecs
        features_space = VectorSpace(dim=2 * self.frame_length)
        features_source = 'features'

        targets_space = VectorSpace(dim=2)
        targets_source = 'targets'

        space = CompositeSpace([features_space, targets_space])
        source = tuple([features_source, targets_source])
        self.data_specs = (space, source)

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('shuffled_sequential')
        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))

    def _validate_source(self, source):
        """
        Verify that all sources in the source tuple are provided by the
        dataset. Raise an error if some requested source is not available.

        Parameters
        ----------
        source : `tuple` of `str`
            Requested sources
        """
        for s in source:
            try:
                self.data_specs[1].index(s)
            except ValueError:
                raise ValueError("the requested source named '" + s + "' " +
                                 "is not provided by the dataset")

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `TIMIT.get()`
            _does_ respect those data specs.
        """
        return self.data_specs

    def get(self, source, indexes):
        """
        .. todo::

            WRITEME
        """
        if type(indexes) is slice:
            indexes = numpy.arange(indexes.start, indexes.stop)
        assert indexes.shape == (1,)
        self._validate_source(source)
        rval = []
        for so in source:
            rval.append(
                self.data[self.data_specs[1].index(so)][indexes]
            )
        return tuple(rval)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        """
        .. todo::

            WRITEME
        """
        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            convert.append(None)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.num_examples, batch_size,
                                          num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

    def _generate_data(self):
        """
        Generates X matrix for DenseDesignMatrix initialization
        function.
        """
        X = []
        y = []

        for init_state, alpha, beta in zip(self.init_state_list,
                                           self.alpha_list,
                                           self.beta_list):
            rval = self._generate_sequence(self.num_samples, init_state, alpha,
                                           beta)
            X.append(rval[0])
            y.append(rval[0])

        return X, y

    def _generate_sequence(self, num_samples, init_state, alpha, beta):
        # Initialize arrays
        X = numpy.zeros((num_samples + 1, 2), dtype=theano.config.floatX)
        X[0, :] = init_state

        # Generate sequence
        for i in range(1, X.shape[0]):
            X[i, 0] = 1 - alpha * X[i - 1, 0] ** 2 + X[i - 1, 1]
            X[i, 1] = beta * X[i - 1, 0]

        return X[:-1, :], X[1:, :]


if __name__ == "__main__":
    dataset = HenonMap(alpha_list=[1.4, 1.3],
                       beta_list=[0.3, 0.2],
                       init_state_list=[numpy.array([0, 0]),
                                        numpy.array([0, 0])],
                       num_samples=1000,
                       rng=None)
    iterator = dataset.iterator(batch_size=1, mode='sequential',
                                data_specs=(VectorSpace(dim=2), 'targets'))
    for y in iterator:
        print y.shape
