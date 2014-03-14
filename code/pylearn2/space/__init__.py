"""
WRITEME
"""
import numpy as np
import theano
from theano import tensor
from theano.gof.op import get_debug_values
from theano.sandbox.cuda.type import CudaNdarrayType
from pylearn2.utils import wraps
from pylearn2.space import _cast, Space, SimplyTypedSpace
from pylearn2.format.target_format import OneHotFormatter


class VectorSequenceSpace(SimplyTypedSpace):
    """
    A space representing a single sequence as a matrix whose first dimension
    represents time and whose second dimension represents the window.

    For now, VectorSequenceSpace cannot be converted to anything else.

    Parameters
    ----------
    max_labels : int
        The number of possible classes/labels. This means that
        all labels should be < max_labels. Example: For MNIST
        there are 10 numbers and hence max_labels = 10.
    dim : int
        The number of indices in one space e.g. for MNIST there is
        one target label and hence dim = 1. If we have an n-gram
        of word indices as input to a neurel net language model, dim = n.
    dtype : str
        A numpy dtype string indicating this space's dtype.
        Must be an integer type e.g. int32 or int64.
    kwargs: passes on to superclass constructor
    """
    def __init__(self, window_dim, dtype='floatX', **kwargs):
        super(VectorSequenceSpace, self).__init__(dtype, **kwargs)

        self.window_dim = window_dim

    def __str__(self):
        """
        Return a string representation.
        """
        return ('%(classname)s(dim=%(dim)s, max_labels=%(max_labels)s, '
                'dtype=%(dtype)s)') % dict(classname=self.__class__.__name__,
                                           dim=self.dim,
                                           max_labels=self.max_labels,
                                           dtype=self.dtype)

    @wraps(Space.__eq__)
    def __eq__(self, other):
        return (type(self) == type(other) and
                self.max_labels == other.max_labels and
                self.dim == other.dim and
                self.dtype == other.dtype)

    @wraps(Space._check_sizes)
    def _check_sizes(self, space):
        if not isinstance(space, VectorSequenceSpace):
            raise ValueError("Can't convert to " + str(space.__class__))
        else:
            if space.window_dim != self.window_dim:
                raise ValueError("Can't convert to VectorSequenceSpace of "
                                 "window_dim %d" %
                                 (space.window_dim,))

    @wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if isinstance(space, VectorSequenceSpace):
            if space.window_dim != self.window_dim:
                raise ValueError("The two VectorSequenceSpaces' window_dim "
                                 "values don't match. This should have been "
                                 "caught by "
                                 "VectorSequenceSpace._check_sizes().")

            return _cast(batch, space.dtype)
        else:
            raise ValueError("Can't convert %s to %s" % (self, space))

    @wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if batch_size is None or batch_size == 1:
            return tensor.matrix(name=name)
        else:
            return ValueError("VectorSequenceSpace does not support batches "
                              "of sequences.")

    @wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        # Only batch size of 1 is supported
        return 1

    @wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        """
        .. todo::

            WRITEME
        """
        # checks that batch isn't a tuple, checks batch.type against self.dtype
        super(VectorSequenceSpace, self)._validate_impl(is_numeric, batch)

        if is_numeric:
            # Use the 'CudaNdarray' string to avoid importing
            # theano.sandbox.cuda when it is not available
            if not isinstance(batch, np.ndarray) \
               and str(type(batch)) != "<type 'CudaNdarray'>":
                raise TypeError("The value of a VectorSequenceSpace batch "
                                "should be a numpy.ndarray, or CudaNdarray, "
                                "but is %s." % str(type(batch)))
            if batch.ndim != 2:
                raise ValueError("The value of a VectorSequenceSpace batch "
                                 "must be 2D, got %d dimensions for %s."
                                 % (batch.ndim, batch))
            if batch.shape[1] != self.window_dim:
                raise ValueError("The width of a VectorSequenceSpace 'batch' "
                                 "must match with the space's window"
                                 "dimension, but batch has window_dim %d and "
                                 "this space's dim is %d."
                                 % (batch.shape[1], self.window_dim))
        else:
            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("VectorSequenceSpace batch should be a theano "
                                "Variable, got " + str(type(batch)))
            if not isinstance(batch.type, (theano.tensor.TensorType,
                                           CudaNdarrayType)):
                raise TypeError("VectorSequenceSpace batch should be "
                                "TensorType or CudaNdarrayType, got " +
                                str(batch.type))
            if batch.ndim != 2:
                raise ValueError("VectorSequenceSpace 'batches' must be 2D, "
                                 "got %d dimensions" % batch.ndim)
            for val in get_debug_values(batch):
                self.np_validate(val)


class IndexSequenceSpace(SimplyTypedSpace):
    """
    A space representing a single sequence as a column whose first dimension
    represents time and whose second dimension represents an index.

    For now, IndexSequenceSpace can only be converted to a VectorSequenceSpace.

    Parameters
    ----------
    max_labels : int
        The number of possible classes/labels. This means that
        all labels should be < max_labels. Example: For MNIST
        there are 10 numbers and hence max_labels = 10.
    dim : int
        The number of indices in one space e.g. for MNIST there is
        one target label and hence dim = 1. If we have an n-gram
        of word indices as input to a neurel net language model, dim = n.
    dtype : str
        A numpy dtype string indicating this space's dtype.
        Must be an integer type e.g. int32 or int64.
    kwargs: passes on to superclass constructor
    """
    def __init__(self, max_labels, window_dim, dtype='int64', **kwargs):
        if not 'int' in dtype:
            raise ValueError("The dtype of IndexSequenceSpace must be an "
                             "integer type")

        super(IndexSequenceSpace, self).__init__(dtype, **kwargs)

        self.max_labels = max_labels
        self.window_dim = window_dim
        self.formatter = OneHotFormatter(self.max_labels)

    def __str__(self):
        """
        Return a string representation.
        """
        return ('%(classname)s(dim=%(window_dim)s, max_labels=%(max_labels)s, '
                'dtype=%(dtype)s)') % dict(classname=self.__class__.__name__,
                                           window_dim=self.window_dim,
                                           max_labels=self.max_labels,
                                           dtype=self.dtype)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.max_labels == other.max_labels and
                self.window_dim == other.window_dim and
                self.dtype == other.dtype)

    @wraps(Space._check_sizes)
    def _check_sizes(self, space):
        if isinstance(space, VectorSequenceSpace):
            # self.max_labels -> merged onehots
            # self.window_dim * self.max_labels -> concatenated
            if space.window_dim not in (self.max_labels,
                                        self.window_dim * self.max_labels):
                raise ValueError("Can't convert to VectorSequenceSpace of "
                                 "window_dim %d. Expected either "
                                 "window_dim=%d (merged one-hots) or %d "
                                 "(concatenated one-hots)" %
                                 (space.dim,
                                  self.max_labels,
                                  self.dim * self.max_labels))
        elif isinstance(space, IndexSequenceSpace):
            if space.window_dim != self.window_dim or space.max_labels != self.max_labels:
                raise ValueError("Can't convert to IndexSequenceSpace of "
                                 "window_dim %d and max_labels %d." %
                                 (space.window_dim, self.max_labels))
        else:
            raise ValueError("Can't convert to " + str(space.__class__))

    @wraps(Space._format_as_impl)
    def _format_as_impl(self, is_numeric, batch, space):
        if isinstance(space, VectorSequenceSpace):
            if self.max_labels == space.window_dim:
                mode = 'merge'
            elif self.window_dim * self.max_labels == space.window_dim:
                mode = 'concatenate'
            else:
                raise ValueError("There is a bug. Couldn't format to a "
                                 "VectorSequenceSpace because it had an "
                                 "incorrect size, but this should've been "
                                 "caught in "
                                 "IndexSequenceSpace._check_sizes().")

            format_func = (self.formatter.format if is_numeric else
                           self.formatter.theano_expr)
            return _cast(format_func(batch, mode=mode), space.dtype)
        elif isinstance(space, IndexSequenceSpace):
            if space.window_dim != self.window_dim or space.max_labels != self.max_labels:
                raise ValueError("The two IndexSequenceSpaces' window_dim and "
                                 "max_labels values don't match. This should "
                                 "have been caught by "
                                 "IndexSequenceSpace._check_sizes().")

            return _cast(batch, space.dtype)
        else:
            raise ValueError("Can't convert %s to %s"
                             % (self, space))

    @wraps(Space.make_theano_batch)
    def make_theano_batch(self, name=None, dtype=None, batch_size=None):
        if batch_size is None or batch_size == 1:
            return tensor.matrix(name=name)
        else:
            return ValueError("IndexSequenceSpace does not support batches "
                              "of sequences.")

    @wraps(Space._batch_size_impl)
    def _batch_size_impl(self, is_numeric, batch):
        # Only batch size of 1 is supported
        return 1

    @wraps(Space._validate_impl)
    def _validate_impl(self, is_numeric, batch):
        """
        .. todo::

            WRITEME
        """
        # checks that batch isn't a tuple, checks batch.type against self.dtype
        super(IndexSequenceSpace, self)._validate_impl(is_numeric, batch)

        if is_numeric:
            # Use the 'CudaNdarray' string to avoid importing
            # theano.sandbox.cuda when it is not available
            if not isinstance(batch, np.ndarray) \
               and str(type(batch)) != "<type 'CudaNdarray'>":
                raise TypeError("The value of a IndexSequenceSpace batch "
                                "should be a numpy.ndarray, or CudaNdarray, "
                                "but is %s." % str(type(batch)))
            if batch.ndim != 2:
                raise ValueError("The value of a IndexSequenceSpace batch "
                                 "must be 2D, got %d dimensions for %s." %
                                 (batch.ndim, batch))
            if batch.shape[1] != self.window_dim:
                raise ValueError("The width of a IndexSequenceSpace batch "
                                 "must match with the space's dimension, but "
                                 "batch has shape %s and window_dim = %d." %
                                 (str(batch.shape), self.window_dim))
        else:
            if not isinstance(batch, theano.gof.Variable):
                raise TypeError("IndexSequenceSpace batch should be a theano "
                                "Variable, got " + str(type(batch)))
            if not isinstance(batch.type, (theano.tensor.TensorType,
                                           CudaNdarrayType)):
                raise TypeError("IndexSequenceSpace batch should be "
                                "TensorType or CudaNdarrayType, got " +
                                str(batch.type))
            if batch.ndim != 2:
                raise ValueError('IndexSequenceSpace batches must be 2D, got '
                                 '%d dimensions' % batch.ndim)
            for val in get_debug_values(batch):
                self.np_validate(val)
