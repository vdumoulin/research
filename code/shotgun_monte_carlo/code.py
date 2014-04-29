#!/usr/bin/env python
__authors__ = ["Vincent Dumoulin"]
__copyright__ = "Copyright 2014"
__credits__ = ["Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
"""
Source code for "A Ballistic Monte Carlo Approximation of pi".

usage: code.py [-h] [-e EXTRACT] [-p PATH] [-v] [-s]

optional arguments:
  -h, --help            show this help message and exit
  -e EXTRACT, --extract EXTRACT
                        path to the directory containing sample images
  -p PATH, --path PATH  path to the directory containing sample images
  -v, --visualize       show figures being created
  -s, --save            save figures to disk
"""
from os import listdir
from os.path import abspath, isfile, join
import argparse
import numpy
from matplotlib import pyplot
from matplotlib import rc
import scipy
from scipy import ndimage


numpy.random.seed(122973)


# Pretty font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)


def generate_synthetic_samples(num_samples=25000):
    """
    Generate a set of synthetic data points.
    
    Samples are normally distributed but bounded in [0, 1].

    Parameters
    ----------
    num_samples : int
        Number of synthetic samples to draw

    Returns
    -------
    samples : numpy.ndarray
        A (num_samples, 2)-sized batch of synthetic samples
    """
    samples = []
    count = 0

    while count < num_samples:
        proposed_sample = numpy.random.normal(loc=0.5, scale=0.3, size=(2,))
        is_accepted = (proposed_sample[0] >= 0 and proposed_sample[0] <= 1 and
                       proposed_sample[1] >= 0 and proposed_sample[1] <= 1)
        if is_accepted:
            samples.append(proposed_sample)
            count += 1

    samples = numpy.asarray(samples)

    return samples


def estimate_pi(samples=None, split=10000, show_results=False,
                save_results=False):
    """
    Return a Monte Carlo estimation of pi.

    The value is obtained by estimating the expected value of g(x, y), where
    x and y are random variables uniformly-distributed in [0, 1] and g(x, y) is
    1 if x^2 + y^2 <= 1 and 0 otherwise.

    This particular expected value can be viewed as the fraction of the area 
    of a unit square occupied by the quarter of a circle of radius 1 centered
    around one of the corners of the square.

    The data will most likely not be uniformly distributed; to compensate for
    that, we use importance sampling by estimating the distribution of the
    samples and weighting them accordingly.

    Parameters
    ----------
    samples : numpy.ndarray, optional
        Data used to estimate the value of pi. Defaults to None, in which case
        synthetic samples are used.
    split : int, optional
        End index of the data used to estimate the PDF. Defaults to 5000.
    show_results : bool, optional
        Whether to display figures on screen. Defaults to `False`.
    save_results : bool, optional
        Whether to save figures on disk. Defaults to `False`.
    """
    # Generate synthetic samples if no samples are provided
    if samples is None:
        print 'Generating synthetic samples...'
        samples = generate_synthetic_samples(num_samples=25000)

    numpy.random.shuffle(samples)

    x = samples[:split, 0]
    y = samples[:split, 1]
    samples_x = samples[split:, 0]
    samples_y = samples[split:, 1]

    g = samples_x ** 2 + samples_y ** 2 <= 1.0
    print 'Estimating sample PDF...'
    densities, x_bins, y_bins, sample_densities = histogram_pdf(x, y,
                                                                samples_x,
                                                                samples_y)

    pi_estimate = 4 * (g / sample_densities).mean()
    pi_error = numpy.abs((pi_estimate - numpy.pi) / numpy.pi)

    # Print the estimate and relevant information
    print "The estimate of pi is %(pi)6.5f, with an error of %(error)4.2f%%" % \
        {"pi": pi_estimate, "error": 100 * pi_error}
    print "Estimation done with %(n)i data points" % {'n': samples_x.shape[0]}

    # Plot the data points along with the quarter circle for reference
    # Plot data points
    pyplot.figure()
    pyplot.axis([0, 1, 0, 1])
    pyplot.xlabel('$x$')
    pyplot.ylabel('$y$')
    pyplot.axes().set_aspect('equal')
    pyplot.scatter(samples_x, samples_y, marker='.', color='0.5',
                   edgecolor='0.5', linewidth=0.25)
    theta = numpy.linspace(start=0.0, stop=numpy.pi/2.0, num=100)
    pyplot.plot(numpy.cos(theta), numpy.sin(theta), color='k',
                linewidth=2.5, linestyle='--')
    if save_results:
        pyplot.savefig('data_points.pdf')
    # Plot PDF
    pyplot.figure()
    pyplot.axis([0, 1, 0, 1])
    pyplot.xlabel('$x$')
    pyplot.ylabel('$y$')
    pyplot.axes().set_aspect('equal')
    pyplot.pcolor(x_bins, y_bins, densities,
                  cmap=pyplot.get_cmap('Greys'))
    pyplot.plot(numpy.cos(theta), numpy.sin(theta), color='k',
                linewidth=2.5, linestyle='--')
    cbar = pyplot.colorbar()
    cbar.ax.set_ylabel('$f(x, y)$')
    if save_results:
        pyplot.savefig('pdf_estimation.pdf')

    if show_results:
        pyplot.show()


def histogram_pdf(x, y, samples_x, samples_y, x_min=0 - 1e-5, x_max=1 + 1e-5,
                  y_min=0 - 1e-5, y_max=1 + 1e-5, min_bins=1, max_bins=50,
                  num_folds=20):
    """
    Estimate the PDF of a set of (x, y) coordinates using k-fold
    cross-validation and compute the likelihood of a set of (samples_x,
    samples_y) coordinates using this estimate

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates of the points used to estimate the PDf
    y : numpy.ndarray
        y-coordinates of the points used to estimate the PDf
    sample_x : numpy.ndarray
        x-coordinates of the points for which to compute the likelihood
    sample_y : numpy.ndarray
        y-coordinates of the points for which to compute the likelihood
    x_min : float, optional
        Lower bound for the x bins of the histogram. Defaults to 0.
    x_max : float, optional
        Upper bound for the x bins of the histogram. Defaults to 1.
    y_min : float, optional
        Lower bound for the y bins of the histogram. Defaults to 0.
    y_max : float, optional
        Upper bound for the y bins of the histogram. Defaults to 1.
    min_bins : int, optional
        Minimal number of bins to try
    max_bins : int, optional
        Maximal number of bins to try
    num_folds : int, optional
        Number of folds for cross-validation

    Returns
    -------
    pdf : numpy.ndarray
    x_bins : numpy.ndarray
    y_bins : numpy.ndarray
    densities : numpy.ndarray
    """
    fold_indexes = []
    fold_length = x.shape[0] / num_folds
    for fold in xrange(num_folds):
        fold_indexes.append((fold * fold_length, (fold + 1) * fold_length))

    optimal_num_bins = None
    optimal_log_likelihood = -numpy.infty
    for num_bins in xrange(min_bins, max_bins + 1):
        valid_densities = []
        for begin_index, end_index in fold_indexes:
            train_x = numpy.concatenate([x[:begin_index], x[end_index + 1:]])
            train_y = numpy.concatenate([y[:begin_index], y[end_index + 1:]])
            valid_x = x[begin_index:end_index]
            valid_y = y[begin_index:end_index]

            pdf, x_bins, y_bins, densities = _histogram_pdf(train_x, train_y,
                                                            x_min, x_max,
                                                            y_min, y_max,
                                                            num_bins)

            x_indexes = (valid_x / x_max * num_bins).astype(int)
            y_indexes = (valid_y / y_max * num_bins).astype(int)
            valid_densities.extend(pdf[[x_indexes, y_indexes]])
        if numpy.min(valid_densities) > 0:
            mean_log_likelihood = numpy.log(valid_densities).mean()
            if mean_log_likelihood > optimal_log_likelihood:
                optimal_log_likelihood = mean_log_likelihood
                optimal_num_bins = num_bins

    print "Optimal number of bins is " + str(optimal_num_bins)
    pdf, x_bins, y_bins, __ = _histogram_pdf(x, y, x_min, x_max, y_min, y_max,
                                             optimal_num_bins)
    x_indexes = (samples_x / x_max * optimal_num_bins).astype(int)
    y_indexes = (samples_y / y_max * optimal_num_bins).astype(int)
    sample_densities = pdf[[x_indexes, y_indexes]]
    return pdf, x_bins, y_bins, sample_densities


def _histogram_pdf(x, y, x_min=0 - 1e-5, x_max=1 + 1e-5, y_min=0 - 1e-5,
                   y_max=1 + 1e-5, num_bins=25):
    """
    Estimate the probability density function of a set of points using the
    2D histogram method and the PDF value for each of those points.

    Parameters
    ----------
    x : `numpy.ndarray`
        x-coordinates of the points
    y : `numpy.ndarray`
        y-coordinates of the points
    x_min : `float`, optional
        Lower bound for the x bins of the histogram. Defaults to 0.
    x_max : `float`, optional
        Upper bound for the x bins of the histogram. Defaults to 1.
    y_min : `float`, optional
        Lower bound for the y bins of the histogram. Defaults to 0.
    y_max : `float`, optional
        Upper bound for the y bins of the histogram. Defaults to 1.
    num_bins : `int`
        Number of bins per dimension. Total number of bins will be
        `num_bins x num_bins`. Defaults to 10.

    Returns
    -------
    pdf : `numpy.ndarray`
    x_bins : `numpy.ndarray`
    y_bins : `numpy.ndarray`
    densities : `numpy.ndarray`
    """
    pdf, x_bins, y_bins = numpy.histogram2d(x, y,
                                            range=((x_min, x_max),
                                                   (y_min, y_max)),
                                            bins=num_bins,
                                            normed=True)

    x_indexes = (x / x_max * num_bins).astype(int)
    y_indexes = (y / y_max * num_bins).astype(int)
    densities = pdf[[x_indexes, y_indexes]]

    return pdf, x_bins, y_bins, densities


def extract_samples(directory_path, show_results=False):
    """
    Extract data points from target scans located in `directory_path`

    Parameters
    ----------
    directory_path : `str`, optional
        Path to the directory containing the images. Defaults to `None`, in
        which case a test image is used.
    show_results : `bool`, optional
        If `True` plot each image overlayed with the data points extracted from
        it. Defaults to `False`.

    Returns
    -------
    samples : `numpy.ndarray`
        All extracted samples
    """
    directory_path = abspath(directory_path)
    samples = []
    paths_list = []
    paths_list.extend([join(directory_path, f) for f in
                       listdir(directory_path) if
                       isfile(join(directory_path, f))
                       and f.endswith('jpg')])

    for path in paths_list:
        # Load image
        image = scipy.misc.imread(path, flatten=True)
        # Set the threshold
        threshold = 50
        # Binarize the image according to the threshold
        binarized_image = image > threshold
        # Find connected components
        labeled_image, num_objects = ndimage.label(binarized_image)
        # Find all centers of mass
        centers_of_mass = numpy.asarray(ndimage.measurements.center_of_mass(
            binarized_image,
            labeled_image,
            numpy.arange(num_objects) + 1
        ))

        # Normalize centers of mass
        normalized_centers_of_mass = numpy.zeros_like(centers_of_mass)
        normalized_centers_of_mass[:, 0] = centers_of_mass[:, 1] / \
                                           image.shape[1]
        normalized_centers_of_mass[:, 1] = 1.0 - centers_of_mass[:, 0] / \
                                           image.shape[0]
        # Add normalized centers of mass to samples
        samples.extend(normalized_centers_of_mass)

        # Optionally plot the centers of mass over the original image for
        # verification puposes
        if show_results:
            x = centers_of_mass[:, 1]
            y = centers_of_mass[:, 0]
            pyplot.figure()
            pyplot.axis([0, image.shape[1], image.shape[0], 0])
            pyplot.imshow(image, cmap=pyplot.get_cmap('gray'))
            pyplot.xlabel('$x$')
            pyplot.ylabel('$y$')
            pyplot.axes().set_aspect('equal')
            pyplot.scatter(x, y, marker='.', color='r', edgecolor='r',
                           linewidth=3)
            pyplot.show()

    # Concatenate all samples into a single numpy array
    samples = numpy.vstack(samples)
    return samples


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extract", default=None,
                        help="path to the directory containing sample images")
    parser.add_argument("-p", "--path", default=None,
                        help="path to the sample file")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="show figures being created")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save figures to disk")
    args = parser.parse_args()

    samples = None
    data_path = args.path
    extract_path = args.extract
    if extract_path is not None:
        extract_samples(directory_path=extract_path,
                        show_results=args.visualize)
    if data_path is not None:
        print 'Retrieving samples...'
        samples = numpy.load(data_path)
    estimate_pi(samples=samples, show_results=args.visualize,
                save_results=args.save)
