"""
fit.py - Do the fitting of the clusters in the image

This takes 7 or 8 arguments:
- Path where the resulting cluster catalog will be saved
- Path to the fits image containing the PSF
- Oversampling factor used when creating the PSF
- Path to the sigma image containing uncertainties for each pixel
- Path to the mask image containing whether or not to use each pixel
- Path to the cleaned cluster catalog
- Size of the fitting region to be used
- Optional argument that must be "ryon_like" if present. If it is present, masking will
  not be done, and the power law slope will be restricted to be greater than 1.
"""
from pathlib import Path
import sys
from collections import defaultdict

from astropy import table, stats
from astropy.io import fits

import numpy as np
from scipy import optimize, special
from tqdm import tqdm

import utils
import fit_utils

# ======================================================================================
#
# Get the parameters the user passed in, load images and catalogs
#
# ======================================================================================
final_catalog = Path(sys.argv[1]).absolute()
psf_path = Path(sys.argv[2]).absolute()
oversampling_factor = int(sys.argv[3])
sigma_image_path = Path(sys.argv[4]).absolute()
mask_image_path = Path(sys.argv[5]).absolute()
cluster_catalog_path = Path(sys.argv[6]).absolute()
snapshot_size = int(sys.argv[7])
if len(sys.argv) > 8:
    if len(sys.argv) != 9 or sys.argv[8] != "ryon_like":
        raise ValueError("Bad list of parameters to fit.py")
    else:
        ryon_like = True
else:
    ryon_like = False

galaxy_name = final_catalog.parent.parent.name
image_data, _, _ = utils.get_drc_image(final_catalog.parent.parent)
psf = fits.open(psf_path)["PRIMARY"].data

# the convolution requires the psf to be normalized, and without any negative values
psf = np.maximum(psf, 0)
psf /= np.sum(psf)

sigma_data = fits.open(sigma_image_path)["PRIMARY"].data
mask_data = fits.open(mask_image_path)["PRIMARY"].data
clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")
# if we're Ryon-like, do no masking, so the mask will just be ones
if ryon_like:
    mask_data = np.ones(mask_data.shape)

snapshot_size_oversampled = snapshot_size * oversampling_factor

# ======================================================================================
#
# Calculate the minimum background level allowed in the fits. This will be 2 sigma
# below the mean sky value (which is calculated here) or the mimum pixel value
# in the cluster snapshot, whichever is lower.
#
# ======================================================================================
flat_im = image_data.flatten()
idxs_nonzero = np.nonzero(image_data)
nonzero_flat_im = image_data[idxs_nonzero]
# calculate some stats
mean_sky, _, sigma_sky = stats.sigma_clipped_stats(nonzero_flat_im, sigma=2.0)
# choose only 2 sigma because the mean is a bit higher than the mode (which should be
# the true sky value). We don't want to go too low because that could let the
# background go too low. We will allow for that if needed as we choose the minimum of
# this value and the minimum pixel value in the cluster snapshot.
background_min = mean_sky - 2 * sigma_sky

# ======================================================================================
#
# Set up the table. We need some dummy columns that we'll fill later
#
# ======================================================================================
def dummy_list_col(n_rows):
    return [[-99.9] * i for i in range(n_rows)]


n_rows = len(clusters_table)
new_cols = [
    "x_pix_single_fitted",
    "y_pix_single_fitted",
    "central_surface_brightness",
    "scale_radius_pixels",
    "axis_ratio",
    "position_angle",
    "power_law_slope",
    "local_background",
]

for col in new_cols:
    clusters_table[col] = dummy_list_col(n_rows)
    clusters_table[col + "_best"] = -99.9

clusters_table["num_boostrapping_iterations"] = -99

# ======================================================================================
#
# Functions to be used in the fitting
#
# ======================================================================================
def calculate_chi_squared(params, cluster_snapshot, error_snapshot, mask):
    """
    Calculate the chi-squared value for a given set of parameters.

    :param params: Tuple of parameters of the EFF profile
    :param cluster_snapshot: Cluster snapshot
    :param error_snapshot: Error snapshot
    :param mask: 2D array used as the mask, that contains 1 where there are pixels to
                 use, and zero where the pixels are not to be used.
    :return:
    """
    _, _, model_snapshot = fit_utils.create_model_image(
        *params, psf, snapshot_size_oversampled, oversampling_factor
    )
    assert model_snapshot.shape == cluster_snapshot.shape
    assert model_snapshot.shape == error_snapshot.shape

    diffs = cluster_snapshot - model_snapshot
    sigma_snapshot = diffs / error_snapshot
    # then use the mask
    sigma_snapshot *= mask
    # then pick the desirec pixels out to be used in the fit
    sum_squared = np.sum(sigma_snapshot ** 2)
    dof = np.sum(mask) - 8
    return sum_squared / dof


def gamma(x, k, theta):
    """
    Gamme distribution PDF: https://en.wikipedia.org/wiki/Gamma_distribution

    :param x: X values to determine the value of the PDF at
    :param k: Shape parameter
    :param theta: Scale parameter
    :param offset: Value at which the distribution starts (other than zero)
    :return: Value of the gamma PDF at this location
    """
    x = np.maximum(x, 0)
    return x ** (k - 1) * np.exp(-x / theta) / (special.gamma(k) * theta ** k)


def gaussian(x, mean, sigma):
    """
    Normal distribution PDF.

    :param x: X values to determine the value of the PDF at
    :param mean: Mean of the Gaussian
    :param sigma: Standard deviation of the Gaussian
    :return: Value of the gamma PDF at this location
    """
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def trapezoid(x, breakpoint):
    """
    PSF (not normalized) of a simple trapezoid shape with one breakpoint.

    Below the breakpoint it is linear to zero, while above it is constant at 1
    to `max_value`

    :param x: X values to determine the value of the PDF at
    :param breakpoint: Value at which the PDF trasitions from the linearly increasing
                       portion to the flat portion
    :param max_value: Maximum allowed value for X. Above this zero will be returned.
    :return: Value of this PDF at this location
    """
    return np.max([np.min([x / breakpoint, 1.0]), 1e-20])


def flat_prior(value, min_value, max_value):
    if value < min_value or value > max_value:
        return 0
    else:
        return 1


def priors(log_mu_0, x_c, y_c, a, q, theta, eta, background):
    """
    Calculate the prior probability for a given model.

    The parameters passed in are the ones for the EFF profile. The parameters are
    treated independently:
    - For the center we use a Gaussian centered on the center of the image with a
      width of 3 image pixels
    - for the scale radius and power law slope we use a Gamma distribution
      with k=1.5, theta=3
    - for the axis ratio we use a simple trapezoid shape, where it's linearly increasing
      up to 0.3, then flat above that.
    All these independent values are multiplied together then returned.

    :return: Total prior probability for the given model.
    """
    prior = 1
    return prior
    # # x and y center have a Gaussian with width of 2 regular pixels, centered on
    # # the center of the snapshot
    # prior *= gamma(a, 1.05, 20)
    # prior *= gamma(eta - 0.6, 1.05, 20)
    # prior *= trapezoid(q, 0.3)
    # # have a minimum allowed value, to stop it from being zero if several of these
    # # parameters are bad.
    # return np.maximum(prior, 1e-50)


def negative_log_likelihood(params, cluster_snapshot, error_snapshot, mask):
    """
    Calculate the negative log likelihood for a model

    We do the negative likelihood becuase scipy likes minimize rather than maximize,
    so minimizing the negative likelihood is maximizing the likelihood

    :param params: Tuple of parameters of the EFF profile
    :param cluster_snapshot: Cluster snapshot
    :param error_snapshot: Error snapshot
    :param mask: 2D array used as the mask, that contains 1 where there are pixels to
                 use, and zero where the pixels are not to be used.
    :return:
    """
    chi_sq = calculate_chi_squared(params, cluster_snapshot, error_snapshot, mask)
    # the exponential gives zeros for very large chi squared values, have a bit of a
    # normalization to correct for that.
    log_data_likelihood = -chi_sq
    log_prior = np.log(priors(*params))
    log_likelihood = log_data_likelihood + log_prior
    assert not np.isnan(log_prior)
    assert not np.isnan(log_data_likelihood)
    assert not np.isinf(log_prior)
    assert not np.isinf(log_data_likelihood)
    assert not np.isneginf(log_prior)
    assert not np.isneginf(log_data_likelihood)
    # return the negative of this so we can minimize this value
    return -log_likelihood


def create_boostrap_mask(original_mask, x_c, y_c):
    """
    Create a temporary mask used during a given bootstrapping iteration.

    We will have two separate regions. Within 9 pixels from the center, we will sample
    on all pixels individually. Outside this central region, we create 3x3 pixel
    boxes. We do bootstrapping on both of these regions independently, then combine
    the selection.

    :param original_mask: Original mask, where other sources can be masked out.
    :param x_c: X center of the cluster
    :param y_c: Y center of the cluster
    :return: Mask that contains the number of times each pixel was selected.
    """
    box_size = 5
    # correct for oversampled pixels
    x_c /= oversampling_factor
    y_c /= oversampling_factor

    # first go through and assign all pixels to either a box or the center. We have a
    # dictionary of lists for the boxes, that will have keys of box location and values
    # of all the pixels in that box.
    outside_boxes = defaultdict(list)
    center_pixels = []
    for x in range(original_mask.shape[1]):
        for y in range(original_mask.shape[0]):
            if original_mask[y, x] == 1:  # only keep pixels not already masked out
                if fit_utils.distance(x, y, x_c, y_c) <= 9:
                    center_pixels.append((x, y))
                else:
                    idx_box_x = x // box_size
                    idx_box_y = y // box_size

                    outside_boxes[(idx_box_x, idx_box_y)].append((x, y))

    # freeze the keys so we can have an order to sample from
    outside_boxes_keys = list(outside_boxes.keys())

    # then we can subsample each of those
    idxs_boxes = np.random.randint(0, len(outside_boxes_keys), len(outside_boxes_keys))
    idxs_center = np.random.randint(0, len(center_pixels), len(center_pixels))

    # then put this into the mask
    temp_mask = np.zeros(original_mask.shape)
    for idx in idxs_boxes:
        key = outside_boxes_keys[idx]
        for x, y in outside_boxes[key]:
            temp_mask[y, x] += 1
    for idx in idxs_center:
        x, y = center_pixels[idx]
        temp_mask[y, x] += 1

    return temp_mask


def fit_model(data_snapshot, uncertainty_snapshot, mask):
    """
    Fits an EFF model to the data passed in, using bootstrapping.

    :param data_snapshot: 2D array holding the pixel values (in units of electrons)
    :param uncertainty_snapshot: 2D array holding the uncertainty in the pixel values,
                                 in units of electrons.
    :param mask: 2D array holding the mask, where 1 is a good pixel, zero is bad.
    :param id_num: The ID of this cluster
    :return: A two-item tuple containing: the fit parameters to all pixels, and the
             history of all parameters took throughout the bootstrapping

    """
    # Create the initial guesses for the parameters
    center = snapshot_size_oversampled / 2.0
    params = (
        np.log10(np.max(data_snapshot) * 3),  # log of peak brightness.
        # Increase that to account for bins, as peaks will be averaged lower.
        center,  # X center in the oversampled snapshot
        center,  # Y center in the oversampled snapshot
        0.5,  # scale radius, in regular pixels. Start small to avoid fitting other things
        1.0,  # axis ratio
        np.pi / 2.0,  # position angle
        5.0,  # power law slope. Start high to give a sharp cutoff and avoid other stuff
        np.min(data_snapshot),  # background
    )

    # some of the bounds are not used because we put priors on them. We don't use priors
    # for the background or max, as these would depend on the individual cluster, so
    # I don't use them. Some are not allowed to be zero, so I don't set zero as the
    # limit, but have a value very close.
    center_half_width = 3 * oversampling_factor
    bounds = [
        # log of peak brightness. The minimum allowed will be the sky value, and the
        # maximum will be 2 orders of magnitude above the first guess.
        (np.log10(np.min(uncertainty_snapshot)), params[0] + 2),
        (center - center_half_width, center + center_half_width,),  # X center
        (center - center_half_width, center + center_half_width,),  # Y center
        (1e-10, None),  # scale radius in regular pixels.
        (1e-10, 1),  # axis ratio
        (None, None),  # position angle
        (0, None),  # power law slope
        # the minimum background allowed will be the smaller of the background level
        # determined above or the minimum pixel value in the shapshot.
        (min(background_min, np.min(data_snapshot)), np.max(data_snapshot)),
    ]
    # modify this in the case of doing things like Ryon, in which case we have a lower
    # limit on the power law slope of 1 (or slightly higher, to avoid overflow errors
    if ryon_like:
        bounds[6] = (1.01, None)

    # first get the results when all good pixels are used, to be used as a starting
    # point when bootstrapping is done, to save time.
    initial_result = optimize.minimize(
        negative_log_likelihood,
        args=(data_snapshot, uncertainty_snapshot, mask),
        x0=params,
        bounds=bounds,
    )

    # Then we do bootstrapping
    n_variables = len(initial_result.x)
    param_history = [[] for _ in range(n_variables)]
    param_std_last = [np.inf for _ in range(n_variables)]

    converge_criteria = 0.5
    converged = [False for _ in range(n_variables)]
    check_spacing = 1
    # larger spacing is more computationally efficient, plots take a while to make, so
    # don't do them that frequently.
    iteration = 0
    while not all(converged):
        iteration += 1

        # make a new mask
        temp_mask = create_boostrap_mask(mask, initial_result.x[1], initial_result.x[2])

        # fit to this selection of pixels
        this_result = optimize.minimize(
            calculate_chi_squared,
            args=(data_snapshot, uncertainty_snapshot, temp_mask),
            x0=params,  # don't use the initial result, to avoid local minima
            bounds=bounds,
        )

        # store the results
        for param_idx in range(n_variables):
            param_history[param_idx].append(this_result.x[param_idx])

        # then check if we're converged
        if iteration % check_spacing == 0:
            for param_idx in range(n_variables):
                # calculate the new standard deviation
                this_std = np.std(param_history[param_idx])
                if this_std == 0:
                    converged[param_idx] = True
                else:  # actually calculate the change
                    last_std = param_std_last[param_idx]
                    diff = abs((this_std - last_std) / this_std)
                    if diff < converge_criteria:
                        converged[param_idx] = True
                    else:
                        converged[param_idx] = False

                # then set the new last value
                param_std_last[param_idx] = this_std

    # then we're done!
    return initial_result.x, np.array(param_history)


# ======================================================================================
#
# Then go through the catalog
#
# ======================================================================================
for row in tqdm(clusters_table):
    # create the snapshot. We use ceiling to get the integer pixel values as python
    # indexing does not include the final value. So when we calcualte the offset, it
    # naturally gets biased low. Moving the center up fixes that in the easiest way.
    x_cen = int(np.ceil(row["x_pix_single"]))
    y_cen = int(np.ceil(row["y_pix_single"]))

    # Get the snapshot, based on the size desired.
    # Since we took the ceil of the center, go more in the negative direction (i.e.
    # use ceil to get the minimum values). This only matters if the snapshot size is odd
    x_min = x_cen - int(np.ceil(snapshot_size / 2.0))
    x_max = x_cen + int(np.floor(snapshot_size / 2.0))
    y_min = y_cen - int(np.ceil(snapshot_size / 2.0))
    y_max = y_cen + int(np.floor(snapshot_size / 2.0))

    data_snapshot = image_data[y_min:y_max, x_min:x_max]
    error_snapshot = sigma_data[y_min:y_max, x_min:x_max]
    mask_snapshot = mask_data[y_min:y_max, x_min:x_max]

    mask_snapshot = fit_utils.handle_mask(
        mask_snapshot, row["x_pix_single"] - x_min, row["y_pix_single"] - y_min
    )

    # then do this fitting!
    results, history = fit_model(data_snapshot, error_snapshot, mask_snapshot)

    # Then add these values to the table
    row["num_boostrapping_iterations"] = len(history[0])

    row["central_surface_brightness_best"] = 10 ** results[0]
    row["x_pix_single_fitted_best"] = x_min + fit_utils.oversampled_to_image(
        results[1], oversampling_factor
    )
    row["y_pix_single_fitted_best"] = y_min + fit_utils.oversampled_to_image(
        results[2], oversampling_factor
    )
    row["scale_radius_pixels_best"] = results[3]
    row["axis_ratio_best"] = results[4]
    row["position_angle_best"] = results[5]
    row["power_law_slope_best"] = results[6]
    row["local_background_best"] = results[7]

    row["central_surface_brightness"] = [10 ** v for v in history[0]]
    row["x_pix_single_fitted"] = [
        x_min + fit_utils.oversampled_to_image(v, oversampling_factor)
        for v in history[1]
    ]
    row["y_pix_single_fitted"] = [
        y_min + fit_utils.oversampled_to_image(v, oversampling_factor)
        for v in history[2]
    ]
    row["scale_radius_pixels"] = history[3]
    row["axis_ratio"] = history[4]
    row["position_angle"] = history[5]
    row["power_law_slope"] = history[6]
    row["local_background"] = history[7]


# ======================================================================================
#
# Then write this output catalog
#
# ======================================================================================
# Before saving we need to put all the columns to the same length, which we can do
# by padding with nans, which are easy to remove
def pad(array, total_length):
    final_array = np.zeros(total_length) * np.nan
    final_array[: len(array)] = array
    return final_array


max_length = max([len(row["central_surface_brightness"]) for row in clusters_table])
for col in new_cols:
    clusters_table[col] = [pad(row[col], max_length) for row in clusters_table]

clusters_table.write(str(final_catalog), format="hdf5", path="table", overwrite=True)
