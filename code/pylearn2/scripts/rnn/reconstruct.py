#!/usr/bin/env python
"""
WRITEME
"""
__authors__ = "Vincent Dumoulin"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.space import CompositeSpace
from research.code.pylearn2.space import VectorSequenceSpace
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
import scipy.io.wavfile as wf
import theano.tensor as T
import numpy
import argparse


def main(model_path):
    print 'Loading model...'
    model = serial.load(model_path)

    dataset_yaml_src = model.dataset_yaml_src
    dataset = yaml_parse.load(dataset_yaml_src)
    data_specs = (CompositeSpace([VectorSequenceSpace(dim=model.nvis),
                                 VectorSequenceSpace(dim=62)]),
                  ('features', 'phones'))
    it = dataset.iterator(mode='sequential', data_specs=data_specs,
                          num_batches=1, batch_size=1)
    original_sequence, phones = it.next()


    X = T.vector('X')
    p = T.vector('p')
    h = T.vector('h')
    out = T.vector('out')

    next_h, pred = model.fprop_step(X, p, h, out)
    fn = theano.function(inputs=[X, p, h, out], outputs=[next_h, pred],
                         on_unused_input='ignore')

    # Reconstruction
    numpy_h = numpy.zeros(model.nhid)
    numpy_out = numpy.zeros(1)
    x_t = numpy.copy(original_sequence[0])

    reconstruction_list = [original_sequence[0]]
    for p_t in phones:
        numpy_h, numpy_out = fn(x_t, p_t, numpy_h, numpy_out)
        reconstruction_list.append(numpy_out)
        x_t[:-1] = x_t[1:]
        x_t[-1] = numpy_out

    numpy_reconstruction = numpy.concatenate(reconstruction_list)
    numpy_reconstruction = numpy_reconstruction * dataset._std + dataset._mean
    numpy_reconstruction = numpy.cast['int16'](numpy_reconstruction)
    wf.write("reconstruction.wav", 16000, numpy_reconstruction)

    # One-on-one prediction
    numpy_h = numpy.zeros(model.nhid)
    numpy_out = numpy.zeros(1)

    prediction_list = [numpy.copy(original_sequence[0])]
    for x_t, p_t in zip(original_sequence, phones):
        numpy_h, numpy_out = fn(x_t, p_t, numpy_h, numpy_out)
        prediction_list.append(numpy_out)

    numpy_prediction = numpy.concatenate(prediction_list)
    numpy_prediction = numpy_prediction * dataset._std + dataset._mean
    numpy_prediction = numpy.cast['int16'](numpy_prediction)
    wf.write("prediction.wav", 16000, numpy_prediction)

    original= numpy.concatenate([original_sequence[0],
                                 original_sequence[1:, -1]])
    original= original * dataset._std + dataset._mean
    original= numpy.cast['int16'](original)
    wf.write("original.wav", 16000, original)


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to the pickled DBM model")
    args = parser.parse_args()

    model_path = args.model_path

    main(model_path)
