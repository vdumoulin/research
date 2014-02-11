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
import scipy
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import CompositeSpace, VectorSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from research.code.scripts.segmentaxis import segment_axis
from research.code.pylearn2.utils.iteration import InfiniteDatasetIterator


class TIMIT(object):
    """
    Base class for the TIMIT dataset. Implements the logic for loading the data
    from disk.
    """
    def _load_data(self, which_set):
        """
        Load the TIMIT data from disk.

        Parameters
        ----------
        which_set : str
            Either "train", "valid" or "test"
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

        # Load data
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


class AcousticTIMIT(DenseDesignMatrix, TIMIT):
    """
    Acoustic-sample-based TIMIT dataset
    """
    _default_seed = (17, 2, 946)

    def __init__(self, which_set, frame_length, start=0, stop=None,
                 rng=_default_seed):
        """
        Parameters
        ----------
        which_set : str
            Either "train", "valid" or "test"
        frame_length : int
            Number of acoustic samples contained in a frame
        start : int, optional
            Index of the Starting 
        rng : object, optional
            A random number generator used for picking random indices into the
            design matrix when choosing minibatches.
        """
        self._load_data(which_set)

        self.frame_length = frame_length

        sequences = self.raw_wav[start:]
        if stop is not None:
            sequences = sequences[:stop]
        segmented_sequences = [segment_axis(sequence, self.frame_length,
                                            self.frame_length - 1)
                               for sequence in sequences]
        data = numpy.vstack(segmented_sequences)
        import pdb; pdb.set_trace()

        # uttfr = [data.frames(z, framelen, overlap) for z in
        #           utterances]
        # fr, ph = zip(*[(x[0], x[1]) for x in uttfr])
        # fr = np.vstack(fr)*2**-15
        # ph = list(itertools.chain(*ph))

        # X = fr[:,0:framelen-1]
        # y = np.array([fr[:,framelen-1]]).T # y.ndim has to be 2
        # if stop is None:
        #     stop = len(y)


class FrameTIMIT(Dataset, TIMIT):
    """
    Frame-based TIMIT dataset
    """
    _default_seed = (17, 2, 946)

    def __init__(self, which_set, frame_length, overlap=0,
                 frames_per_example=1, rng=_default_seed):
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
        # Load data from disk
        self._load_data(which_set)

        self.frame_length = frame_length
        self.overlap = overlap
        self.frames_per_example = frames_per_example

        # Transform data in DenseDesignMatrix format
        visiting_order = []
        segmented_sequences = []
        segmented_phonemes = []
        segmented_words = []
        segmented_speaker_info = []
        for i, sequence in enumerate(self.raw_wav):
            # Get the phonemes
            phonemes_list_start = self.sequences_to_phonemes[i][0]
            phonemes_list_end = self.sequences_to_phonemes[i][1]
            phonemes_sequence = self.phonemes[phonemes_list_start:phonemes_list_end]
            encoded_phonemes_sequence = numpy.zeros_like(sequence)
            # Some timestamp does not correspond to any phoneme so 0 is
            # the index for "NO_PHONEME" and the other index are shifted by one
            for (phoneme_start, phoneme_end, phoneme) in phonemes_sequence:
                encoded_phonemes_sequence[phoneme_start:phoneme_end] = phoneme + 1

            # Get the words
            words_list_start = self.sequences_to_words[i][0]
            words_list_end = self.sequences_to_words[i][1]
            words_sequence = self.words[words_list_start:words_list_end]
            encoded_words_sequence = numpy.zeros_like(sequence)
            # Some timestamp does not correspond to any word so 0 is
            # the index for "NO_WORD" and the other index are shifted by one
            for (word_start, word_end, word) in words_sequence:
                encoded_words_sequence[word_start:word_end] = word + 1

            # Binary variable announcing the end of the word or phoneme
            end_phoneme = numpy.zeros_like(encoded_phonemes_sequence)
            end_word = numpy.zeros_like(encoded_words_sequence)

            for j in range(len(encoded_phonemes_sequence) - 1):
                if encoded_phonemes_sequence[j] != encoded_phonemes_sequence[j+1]:
                    end_phoneme[j] = 1
                if encoded_words_sequence[j] != encoded_words_sequence[j+1]:
                    end_word[j] = 1

            end_phoneme[-1] = 1
            end_word[-1] = 1

            # Find the speaker id
            speaker_id = self.speaker_id[i]
            # Find the speaker info
            speaker_info = self.speaker_info_list[speaker_id]

            # Segment sequence
            segmented_sequence = segment_axis(sequence, self.frame_length,
                                              self.overlap)

            # Take the most occurring phoneme in a sequence
            segmented_encoded_phonemes_sequence = segment_axis(
                encoded_phonemes_sequence,
                self.frame_length,
                self.overlap
            )
            segmented_encoded_phonemes_sequence = scipy.stats.mode(
                segmented_encoded_phonemes_sequence,
                axis=1
            )[0].flatten()
            segmented_encoded_phonemes_sequence = numpy.asarray(
                segmented_encoded_phonemes_sequence, dtype='int'
            )

            # Take the most occurring word in a sequence
            segmented_encoded_words_sequence = segment_axis(
                encoded_words_sequence, self.frame_length, self.overlap
            )
            segmented_encoded_words_sequence = scipy.stats.mode(
                segmented_encoded_words_sequence,
                axis=1
            )[0].flatten()
            segmented_encoded_words_sequence = numpy.asarray(
                segmented_encoded_words_sequence,
                dtype='int'
            )

            # Announce the end if and only if it was announced in the current frame
            end_phoneme = segment_axis(end_phoneme, self.frame_length,
                                       self.overlap)
            end_phoneme = end_phoneme.max(axis=1)
            end_word = segment_axis(end_word, self.frame_length, self.overlap)
            end_word = end_word.max(axis=1)

            segmented_sequences.append(segmented_sequence)

            self.raw_wav[i] = segmented_sequence

            for j in xrange(0, segmented_sequence.shape[0] - self.frames_per_example):
                visiting_order.append((i, j))

        self.visiting_order = visiting_order

        # DataSpecs
        X_space = VectorSpace(dim=self.frame_length * self.frames_per_example)
        X_source = 'features'
        y_space = VectorSpace(dim=self.frame_length)
        y_source = 'targets'
        space = CompositeSpace((X_space, y_space))
        source = (X_source, y_source)
        self.data_specs = (space, source)

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('sequential')
        self._iter_data_specs = self.data_specs

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs

    def get_visiting_order(self):
        return self.visiting_order

    def get(self, index_tuple_list):
        if type(index_tuple_list) is not list:
            index_tuple_list = [index_tuple_list]

        X = []
        y = []
        for sequence_id, index in index_tuple_list:
            X.append(self.raw_wav[sequence_id][index:index +
                                               self.frames_per_example].flatten())
            y.append(self.raw_wav[sequence_id][index + self.frames_per_example])
        return numpy.array(X), numpy.array(y)

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
        return InfiniteDatasetIterator(self,
                                       mode(len(self.visiting_order),
                                            batch_size, num_batches, rng),
                                       data_specs=data_specs,
                                       return_tuple=return_tuple,
                                       convert=convert)


if __name__ == "__main__":
    timit = AcousticTIMIT("valid", frame_length=200)
