"""
Pylearn2 wrapper for the TIMIT dataset
"""
__authors__ = ["Vincent Dumoulin"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh", "Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
__email__ = "dumouliv@iro"


import os.path
import functools
import numpy
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.datasets.dataset import Dataset
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from research.code.scripts.segmentaxis import segment_axis
from research.code.pylearn2.utils.iteration import FiniteDatasetIterator
import scipy.stats


class TIMIT(Dataset):
    """
    Frame-based TIMIT dataset
    """
    _default_seed = (17, 2, 946)

    def __init__(self, which_set, frame_length, overlap=0,
                 frames_per_example=1, start=0, stop=None, rng=_default_seed):
        """
        Parameters
        ----------
        which_set : str
            Either "train", "valid" or "test"
        frame_length : int
            Number of acoustic samples contained in a frame
        overlap : int, optional
            Number of overlapping acoustic samples for two consecutive frames.
            Defaults to 0, meaning frames don't overlap.
        frames_per_example : int, optional
            Number of frames in a training example. Defaults to 1.
        rng : object, optional
            A random number generator used for picking random indices into the
            design matrix when choosing minibatches.
        """
        self.frame_length = frame_length
        self.overlap = overlap
        self.frames_per_example = frames_per_example
        self.offset = self.frame_length - self.overlap
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = numpy.random.RandomState(rng)

        # Load data from disk
        self._load_data(which_set)

        if stop is not None:
            self.raw_wav = self.raw_wav[start:stop]
        else:
            self.raw_wav = self.raw_wav[start:]

        features_map = []
        targets_map = []
        phones_map = []
        words_map = []

        n_seq = len(self.raw_wav)
        self.phn_seq = []
        self.wrd_seq = []
        for sequence_id in range(len(self.raw_wav)):
            # Get the phonemes
            phn_l_start = self.sequences_to_phonemes[sequence_id][0]
            phn_l_end = self.sequences_to_phonemes[sequence_id][1]
            phn_start_end = self.phonemes[phn_l_start:phn_l_end]
            phn_sequence = numpy.zeros(len(self.raw_wav[sequence_id]))
            # Some timestamp does not correspond to any phoneme so 0 is 
            # the index for "NO_PHONEME" and the other index are shifted by one
            for (phn_start, phn_end, phn) in phn_start_end:
                phn_sequence[phn_start:phn_end] = phn+1

            phn_segmented_sequence = segment_axis(phn_sequence, frame_length, overlap)
            phn_segmented_sequence = scipy.stats.mode(phn_segmented_sequence, axis=1)[0].flatten()
            phn_segmented_sequence = numpy.asarray(phn_segmented_sequence, dtype='int')
            self.phn_seq.append(phn_segmented_sequence)

            # Get the words
            wrd_l_start = self.sequences_to_words[sequence_id][0]
            wrd_l_end = self.sequences_to_words[sequence_id][1]
            wrd_start_end = self.words[wrd_l_start:wrd_l_end]
            wrd_sequence = numpy.zeros(len(self.raw_wav[sequence_id]))
            # Some timestamp does not correspond to any word so 0 is 
            # the index for "NO_WORD" and the other index are shifted by one
            for (wrd_start, wrd_end, wrd) in wrd_start_end:
                wrd_sequence[wrd_start:wrd_end] = wrd+1

            wrd_segmented_sequence = segment_axis(wrd_sequence, frame_length, overlap)
            wrd_segmented_sequence = scipy.stats.mode(wrd_segmented_sequence, axis=1)[0].flatten()
            wrd_segmented_sequence = numpy.asarray(wrd_segmented_sequence, dtype='int')
            self.wrd_seq.append(wrd_segmented_sequence)

        self.phn_seq = numpy.array(self.phn_seq)
        self.wrd_seq = numpy.array(self.wrd_seq)

        for sequence_id, sequence in enumerate(self.raw_wav):
            segmented_sequence = segment_axis(sequence, frame_length, overlap)
            self.raw_wav[sequence_id] = segmented_sequence

            num_frames = segmented_sequence.shape[0]
            num_examples = num_frames - self.frames_per_example
            for example_id in xrange(num_examples):
                features_map.append([sequence_id, example_id,
                                     example_id + self.frames_per_example])
                targets_map.append([sequence_id,
                                    example_id + self.frames_per_example])
                phones_map.append([sequence_id, example_id])
                words_map.append([sequence_id, example_id])

        features_map = numpy.asarray(features_map)
        targets_map = numpy.asarray(targets_map)
        phones_map = numpy.asarray(phones_map)
        words_map = numpy.asarray(words_map)

        self.num_examples = features_map.shape[0]

        # DataSpecs
        features_space = VectorSpace(
            dim=self.frame_length * self.frames_per_example
        )
        features_source = 'features'
        features_dtype = self.raw_wav[0].dtype
        features_map_fn = lambda indexes: [
            self.raw_wav[index[0]][index[1]:index[2]].ravel()
            for index in features_map[indexes]
        ]

        targets_space = VectorSpace(dim=self.frame_length)
        targets_source = 'targets'
        targets_dtype = self.raw_wav[0].dtype
        targets_map_fn = lambda indexes: [
            self.raw_wav[index[0]][index[1]]
            for index in targets_map[indexes]
        ]

        phones_space = VectorSpace(dim=1)
        phones_source = 'phones'
        phones_dtype = self.phn_seq[0].dtype
        phones_map_fn = lambda indexes: [
            self.phn_seq[index[0]][index[1]]
            for index in phones_map[indexes]
        ]

        words_space = VectorSpace(dim=1)
        words_source = 'words'
        words_dtype = self.wrd_seq[0].dtype
        words_map_fn = lambda indexes: [
            self.wrd_seq[index[0]][index[1]]
            for index in words_map[indexes]
        ]

        space = CompositeSpace((features_space, targets_space, phones_space,
                                words_space))
        source = (features_source, targets_source, phones_source, words_source)
        self.data_specs = (space, source)
        self.dtypes = (features_dtype, targets_dtype, phones_dtype,
                       words_dtype)
        self.map_functions = (features_map_fn, targets_map_fn, phones_map_fn,
                              words_map_fn)

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('shuffled_sequential')
        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))

    def _load_data(self, which_set):
        """
        Load the TIMIT data from disk.

        Parameters
        ----------
        which_set : str
            Subset of the dataset to use (either "train", "valid" or "test")
        """
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")

        # Create file paths
        timit_base_path = os.path.join(os.environ["PYLEARN2_DATA_PATH"],
                                       "timit/readable")
        speaker_info_list_path = os.path.join(timit_base_path, "spkrinfo.npy")
        phonemes_list_path = os.path.join(timit_base_path,
                                          "reduced_phonemes.pkl")
        words_list_path = os.path.join(timit_base_path, "words.pkl")
        speaker_features_list_path = os.path.join(timit_base_path,
                                                  "spkr_feature_names.pkl")
        speaker_id_list_path = os.path.join(timit_base_path,
                                            "speakers_ids.pkl")
        raw_wav_path = os.path.join(timit_base_path, which_set + "_x_raw.npy")
        phonemes_path = os.path.join(timit_base_path,
                                     which_set + "_redux_phn.npy")
        sequences_to_phonemes_path = os.path.join(timit_base_path,
                                                  which_set +
                                                  "_seq_to_phn.npy")
        words_path = os.path.join(timit_base_path, which_set + "_wrd.npy")
        sequences_to_words_path = os.path.join(timit_base_path,
                                               which_set + "_seq_to_wrd.npy")
        speaker_path = os.path.join(timit_base_path,
                                    which_set + "_spkr.npy")

        # Load data. For now most of it is not used, as only the acoustic
        # samples are provided, but this is bound to change eventually.
        # Global data
        self.speaker_info_list = serial.load(
            speaker_info_list_path
        ).tolist().toarray()
        self.speaker_id_list = serial.load(speaker_id_list_path)
        self.speaker_features_list = serial.load(speaker_features_list_path)
        self.words_list = serial.load(words_list_path)
        self.phonemes_list = serial.load(phonemes_list_path)
        # Set-related data
        self.raw_wav = serial.load(raw_wav_path)
        self.phonemes = serial.load(phonemes_path)
        self.sequences_to_phonemes = serial.load(sequences_to_phonemes_path)
        self.words = serial.load(words_path)
        self.sequences_to_words = serial.load(sequences_to_words_path)
        self.speaker_id = numpy.asarray(serial.load(speaker_path), 'int')

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

    def dtype_of(self, source):
        """
        Returns the dtype of the requested source
        """
        self._validate_source(source)
        return tuple(self.dtypes[self.data_specs[1].index(so)]
                     for so in source)

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
        self._validate_source(source)
        return tuple(
            numpy.asarray(
                self.map_functions[self.data_specs[1].index(so)](indexes)
            )
            for so in source
        )

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


if __name__ == "__main__":
    timit = TIMIT("valid", frame_length=240, overlap=10, frames_per_example=5)
    it = timit.iterator(mode='shuffled_sequential', batch_size=2000)
    for (f, t) in it:
        print f.shape
