import numpy
from research.code.pylearn2.datasets.timit import format_sequences


def test_format_two_sequences_no_overlap():
    x = [numpy.arange(20), numpy.arange(23)]
    frame_length = 5
    overlap = 0
    frames_per_example = 2

    segmented_x, features_map, targets_map = format_sequences(
        sequences=x,
        frame_length=frame_length,
        overlap=overlap,
        frames_per_example=frames_per_example
    )

    # Check for correctness of segmented sequences array
    assert numpy.equal(segmented_x, numpy.array(
        [[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19],
         [ 0,  0,  0,  0,  0],
         [ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19],
         [ 0,  0,  0,  0,  0]]
    )).all()

    # Check for correctness of features map array
    assert numpy.equal(
        features_map,
        numpy.array([[0, 2], [1, 3], [5, 7], [6, 8]])
    ).all()

    # Check for correctness of targets map array
    assert numpy.equal(
        targets_map,
        numpy.array([2, 3, 7, 8])
    ).all()


def test_format_two_sequences_with_overlap():
    x = [numpy.arange(20), numpy.arange(23)]
    frame_length = 5
    overlap = 2
    frames_per_example = 2

    segmented_x, features_map, targets_map = format_sequences(
        sequences=x,
        frame_length=frame_length,
        overlap=overlap,
        frames_per_example=frames_per_example
    )

    # Check for correctness of segmented sequences array
    assert numpy.equal(segmented_x, numpy.array(
        [[ 0,  1,  2,  3,  4],
         [ 3,  4,  5,  6,  7],
         [ 6,  7,  8,  9, 10],
         [ 9, 10, 11, 12, 13],
         [12, 13, 14, 15, 16],
         [15, 16, 17, 18, 19],
         [18, 19,  0,  0,  1],
         [ 0,  1,  2,  3,  4],
         [ 3,  4,  5,  6,  7],
         [ 6,  7,  8,  9, 10],
         [ 9, 10, 11, 12, 13],
         [12, 13, 14, 15, 16],
         [15, 16, 17, 18, 19],
         [18, 19, 20, 21, 22]]
    )).all()

    # Check for correctness of features map array
    assert numpy.equal(
        features_map,
        numpy.array([[0, 2], [1, 3], [2, 4], [3, 5],
                     [7, 9], [8, 10], [9, 11], [10, 12], [11, 13]])
    ).all()

    # Check for correctness of targets map array
    assert numpy.equal(
        targets_map,
        numpy.array([2, 3, 4, 5, 9, 10, 11, 12, 13])
    ).all()


def test_format_three_sequences_no_overlap():
    x = [numpy.arange(13), numpy.arange(16), numpy.arange(11)]
    frame_length = 5
    overlap = 0
    frames_per_example = 1

    segmented_x, features_map, targets_map = format_sequences(
        sequences=x,
        frame_length=frame_length,
        overlap=overlap,
        frames_per_example=frames_per_example
    )

    # Check for correctness of segmented sequences array
    assert numpy.equal(segmented_x, numpy.array(
        [[ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [ 0,  0,  0,  0,  0],
         [ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [10, 11, 12, 13, 14],
         [ 0,  0,  0,  0,  0],
         [ 0,  1,  2,  3,  4],
         [ 5,  6,  7,  8,  9],
         [ 0,  0,  0,  0,  0]]
    )).all()

    # Check for correctness of features map array
    assert numpy.equal(
        features_map,
        numpy.array([[0, 1], [3, 4], [4, 5], [7, 8]])
    ).all()

    # Check for correctness of targets map array
    assert numpy.equal(
        targets_map,
        numpy.array([1, 4, 5, 8])
    ).all()
