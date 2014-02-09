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
import cPickle
import numpy
from pylearn2.utils import serial
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.datasets.dataset import Dataset
from research.code.scripts.segmentaxis import segment_axis


class TIMIT(Dataset):
    """
    TIMIT dataset
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
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")
        
        self.frame_length = frame_length
        self.overlap = overlap
        self.frames_per_example = frames_per_example

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
        self.speaker_info_list = serial.load(speaker_info_list_path).tolist().toarray()
        self.speaker_id_list = serial.load(speaker_id_list_path)
        self.speaker_features_list = serial.load(speaker_features_list_path)
        self.words_list = serial.load(words_list_path)
        self.phonemes_list = serial.load(phonemes_list_path)
        self.raw_wav = serial.load(raw_wav_path)
        self.phonemes = serial.load(phonemes_path) 
        self.sequences_to_phonemes = serial.load(sequences_to_phonemes_path)
        self.words = serial.load(words_path) 
        sequences_to_words = serial.load(sequences_to_words_path)
        speaker_id = numpy.asarray(serial.load(speaker_path), 'int')

        # Transform data in DenseDesignMatrix format
        visiting_order = []
        for i, sequence in enumerate(self.raw_wav):
            segmented_sequence = segment_axis(sequence, self.frame_length,
                                              self.overlap)
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

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs


if __name__ == "__main__":
    timit = TIMIT("train", frame_length=20, overlap=10, frames_per_example=4)
