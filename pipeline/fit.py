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
# if we're Ryon-like, do no masking, so the mask will just be -2 (the value for no
# masked region)
if ryon_like:
    mask_data = np.ones(mask_data.shape) * -2

snapshot_size_oversampled = snapshot_size * oversampling_factor

# set up the background scale factor. We do this as the range of possible values for the
# background is quite large, so it's hard for the fit to cover this whole range. In
# addition, in the gradient calculation, very small changes in the background have
# basically no change in the fit quality. To alleviate these issues, we simple scale
# the background value used in the fit. We divide it by some large number. This simple
# fix solves both issues. Using the log would also work, but is harder since we have
# negative numbers. The position angle actually has similar issues, so we do the similar
# thing
bg_scale_factor = 1e3
theta_scale_factor = 1e2

# ======================================================================================
#
# Set up the table. We need some dummy columns that we'll fill later
#
# ======================================================================================
def dummy_list_col(n_rows):
    return [[-99.9] * i for i in range(n_rows)]


n_rows = len(clusters_table)
new_cols = [
    "x_fitted",
    "y_fitted",
    "x_pix_snapshot_oversampled",
    "y_pix_snapshot_oversampled",
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
clusters_table["success"] = -99

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
    # then use the mask and the weights
    sigma_snapshot *= mask
    # do the radial weighting. Need to get the data coordinates of the center
    sigma_snapshot *= fit_utils.radial_weighting(
        cluster_snapshot,
        fit_utils.oversampled_to_image(params[1], oversampling_factor),
        fit_utils.oversampled_to_image(params[2], oversampling_factor),
        style="annulus",
    )
    return np.sum(np.abs(sigma_snapshot))


def estimate_background(data, mask, x_c, y_c, min_radius):
    """
    Estimate the true background value.

    This will be defined to be the median of all pixels beyond min_radius

    :param data: Values at each pixel
    :param x_c: X coordinate of the center, in coordinates of the sigmas snapshot
    :param y_c: Y coordinate of the center, in coordinates of the sigmas snapshot
    :param min_radius: Minimum radius to include in the calculation
    :return: median(pixel_value) and std(pixel_value) where r > min_radius,
    """
    good_bg = []
    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            if fit_utils.distance(x, y, x_c, y_c) > min_radius and mask[y, x] > 0:
                good_bg.append(data[y, x])

    if len(good_bg) > 0:
        mean, median, std = stats.sigma_clipped_stats(
            good_bg, sigma_lower=None, sigma_upper=3.0, maxiters=None,
        )
        return mean, std
    else:
        return np.min(data), np.inf


def mean_of_normal(x_value, y_value, sigma, above):
    """
    Calculate the mean required for a normal distribution with a given width to take
    the given y value at a specific x value.

    :param x_value: Value at which the normal distribution takes a value of `y_value`.
    :param x_value: Value of the normal distribution at `x_value`
    :param above: Whether `x_value` should be a greater value than the returned mean
                  or not. This is needed since there are two roots, placing the normal
                  distribution lower or higher than the value paseed in.
    :return: The mean of normal distribution matching the properties above.
    """
    if y_value > 1:
        raise ValueError("Invalid y_value in `center_of_normal`")
    # This math can be worked out by hand
    # mean = x +- sqrt(ln(y^-2))
    second_term = sigma * np.sqrt(np.log(y_value ** (-2)))
    if above:
        second_term *= -1
    return x_value + second_term


def log_of_normal(x, mean, sigma):
    """
    Log of the normal distribution PDF. This is normalized to be 0 at the mean.

    :param x: X values to determine the value of the PDF at
    :param mean: Mean of the Gaussian
    :param sigma: Standard deviation of the Gaussian
    :return: natural log of the normal PDF at this location
    """
    return -0.5 * ((x - mean) / sigma) ** 2


def flat_normal_edge(x, edge, side_width, boundary_value, side):
    """
    Probability density function that is flat with value at 1, but with a lognormal
    PDF with fixed width on one side. This returns the log of that density.

    This is useful as it is continuous and smooth at the boundaries of the
    flat region.

    :param x: Value to determine the value of the PDF at.
    :param edge: Lower or upper value where the PDF beging to break
    :param side_log_width: Gaussian width (in dex) of the normal distribution
                           used for the region on the side.
    :param boundary_value: The value that the pdf should take at the edge. If this is
                           1.0, the flat region will extend to the edges. However, if
                           another is value is preferred at those edges, the flat
                           region will shrink to make room for that.
    :param side: whether the lognormal size is on the "lower" or "upper" side
    :return:
    """
    above = side == "upper"
    mean = mean_of_normal(edge, boundary_value, side_width, above)

    # check that we did the math right for the shape of the distribution
    assert np.isclose(np.exp(log_of_normal(edge, mean, side_width)), boundary_value)

    if (side == "upper" and x <= mean) or (side == "lower" and x >= mean):
        return 0
    else:
        return log_of_normal(x, mean, side_width)


def logistic(x, minimum, maximum, x_0, scale):
    """
    Generalized logistic function that goes from some min to some max

    :param x: Value at which to evaluate the logistic function
    :param minimum: Asymptotic value for low values of x
    :param maximum: Asymptotic value for high values of x
    :param x_0: Central value at which the transition happens
    :param scale: Scale factor for the width of the transition
    :return: Value of the logistic function at this value.
    """
    height = maximum - minimum
    return minimum + height / (1 + np.exp((x_0 - x) / scale))


def log_priors(
    log_mu_0, x_c, y_c, a, q, theta, eta, background, estimated_bg, estimated_bg_sigma
):
    """
    Calculate the prior probability for a given model.

    :param estimated_bg: the estimated background value, to be used as the mean of the
                         Gaussian prior on the background.
    :param estimated_bg_sigma: the scatter in the estimated background. The sigma of the
                               Gaussian prior on the background will be 0.1 times this.


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
    log_prior = 0
    # prior are multiplicative, or additive in log space
    # have the low prior be on the minor axis
    # log_prior += flat_normal_edge(np.log10(a), np.log10(0.1), 0.1, 1.0, "lower")
    # # then the upper prior on the major axis
    # log_prior += flat_normal_edge(np.log10(a), np.log10(15), 0.1, 1.0, "upper")
    # then a straight prior on the scale radius
    log_prior += flat_normal_edge(q, 0.3, 0.1, 1.0, "lower")
    # the width of the prior on the background depends on the value of the power law
    # slope. Below 1 it will be strict (0.1 sigma), as this is when we have issues with
    # estimating the background, while for higher values of eta the background prior
    # will be less strict. We want a smooth transition of this width, as any sharp
    # transition will give artifacts in the resulting distributions.
    width = logistic(eta, 0.1, 1.0, 1.0, 0.1) * estimated_bg_sigma
    log_prior += log_of_normal(background, estimated_bg, width)
    return log_prior


def negative_log_likelihood(
    params, cluster_snapshot, error_snapshot, mask, estimated_bg, estimated_bg_sigma
):
    """
    Calculate the negative log likelihood for a model

    We do the negative likelihood becuase scipy likes minimize rather than maximize,
    so minimizing the negative likelihood is maximizing the likelihood

    :param params: Tuple of parameters of the EFF profile
    :param cluster_snapshot: Cluster snapshot
    :param error_snapshot: Error snapshot
    :param mask: 2D array used as the mask, that contains 1 where there are pixels to
                 use, and zero where the pixels are not to be used.
    :param estimated_bg: the estimated background value, to be used as the mean of the
                         Gaussian prior on the background.
    :param estimated_bg_sigma: the scatter in the estimated background. The sigma of the
                               Gaussian prior on the background will be 0.1 times this.
    :return:
    """
    # postprocess them from the beginning to make things simpler. This allows the
    # fitting machinery to do what it wants, but under the hood we only use
    # reasonable parameter value. This also handles the background scaling.
    params = postprocess_params(*params)

    chi_sq = calculate_chi_squared(params, cluster_snapshot, error_snapshot, mask)
    log_data_likelihood = -chi_sq / 2.0
    # Need to postprocess the parameters before calculating the prior, as the prior
    # is on the physically reasonable values, we need to make sure that's correct.
    log_prior = log_priors(*params, estimated_bg, estimated_bg_sigma)
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


def postprocess_params(log_mu_0, x_c, y_c, a, q, theta, eta, background):
    """
    Postprocess the parameters, namely the axis ratio and position angle.

    This is needed since we let the fit have axis ratios larger than 1, and position
    angles of any value. Axis ratios larger than 1 indicate that we need to flip the
    major and minor axes. This requires rotating the position angle 90 degrees, and
    shifting the value assigned to the major axis to correct for the improper axis
    ratio.
    """
    # handle background
    background *= bg_scale_factor
    theta *= theta_scale_factor

    # q and a can be negative, fix that before any further processing
    a = abs(a)
    q = abs(q)
    if q > 1.0:
        q_final = 1.0 / q
        a_final = a * q
        theta_final = (theta - (np.pi / 2.0)) % np.pi
        return log_mu_0, x_c, y_c, a_final, q_final, theta_final, eta, background
    else:
        return log_mu_0, x_c, y_c, a, q, theta % np.pi, eta, background


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
    data_center = snapshot_size / 2.0
    # estimate the fixed quantity for the background
    estimated_bg, bg_scatter = estimate_background(
        data_snapshot, mask, data_center, data_center, 6
    )

    # get the center pixel coordinate in the oversampled snapshot
    center = snapshot_size_oversampled / 2.0
    # Create the initial guesses for the parameters
    params = (
        np.log10(np.max(data_snapshot) * 3),  # log of peak brightness.
        # Increase that to account for bins, as peaks will be averaged lower.
        center,  # X center in the oversampled snapshot
        center,  # Y center in the oversampled snapshot
        0.5,  # scale radius, in regular pixels. Start small to avoid fitting other things
        0.8,  # axis ratio
        0.0,  # position angle
        2.5,  # power law slope. Start high to give a sharp cutoff and avoid other stuff
        estimated_bg / bg_scale_factor,  # background
    )

    # some of the bounds are not used because we put priors on them. We don't use priors
    # for the background or max, as these would depend on the individual cluster, so
    # I don't use them. Some are not allowed to be zero, so I don't set zero as the
    # limit, but have a value very close. We allow axis ratios greater than 1 to make
    # the fitting routine have more flexibility. For example, if the position angle is
    # correct but it's aligned with what should be the minor axis instead of the major,
    # the axis ratio can go above one to fix that issue. We then have to process things
    # afterwards to flip it back, but that's not an issue. We do similar things with
    # allowing the axis ratio and scale radius to be negative. They get squared in the
    # EFF profile anyway, so their sign doesn't matter. They are singular at a=0 and
    # q=0, but hopefully floating point values will stop that from being an issue
    center_half_width = 2 * oversampling_factor
    bounds = [
        (None, 100),  # log of peak brightness.
        (center - center_half_width, center + center_half_width),  # X center
        (center - center_half_width, center + center_half_width),  # Y center
        (None, None),  # scale radius in regular pixels.
        (None, None),  # axis ratio
        (None, None),  # position angle
        (0, None),  # power law slope
        (None, None),  # background
    ]

    # set some of the convergence criteria parameters for the scipy  minimize routine
    ftol = 2e-9  # ftol is roughly the default value
    gtol = 1e-7  # stopping value of the gradient, 100x lower than default value
    eps = 1e-8  # absolute step size for the gradient calculation, default
    maxfun = np.inf  # max number of function evaluations
    maxiter = np.inf  # max number of iterations
    maxls = 200  # max line search steps per iteration. Default is 20

    # first get the results when all good pixels are used, to be used as a starting
    # point when bootstrapping is done, to save time.
    initial_result_struct = optimize.minimize(
        negative_log_likelihood,
        args=(data_snapshot, uncertainty_snapshot, mask, estimated_bg, bg_scatter),
        x0=params,
        bounds=bounds,
        method="L-BFGS-B",  # default method when using bounds
        options={
            "ftol": ftol,
            "gtol": gtol,
            "eps": eps,
            "maxfun": maxfun,
            "maxiter": maxiter,
            "maxls": maxls,
        },
    )

    success = initial_result_struct.success
    # postprocess these parameters
    initial_result = postprocess_params(*initial_result_struct.x)

    # Then we do bootstrapping
    n_variables = len(initial_result)
    param_history = [[0.7] for _ in range(n_variables)]
    param_std_last = [np.inf for _ in range(n_variables)]

    converge_criteria = 0.5  # fractional change in std required for convergence
    converged = [False for _ in range(n_variables)]
    check_spacing = 1  # how many iterations between checking the std
    iteration = 0
    while False:  # not all(converged):
        iteration += 1

        # make a new mask
        temp_mask = create_boostrap_mask(mask, initial_result[1], initial_result[2])

        # fit to this selection of pixels
        this_result_struct = optimize.minimize(
            calculate_chi_squared,
            args=(data_snapshot, uncertainty_snapshot, temp_mask),
            # use the best fit results as the initial guess, to get the uncertainty
            # around that value. This should also reduce the time needed to converge
            x0=initial_result,
            bounds=bounds,
            method="L-BFGS-B",  # default method when using bounds
            options={
                "ftol": ftol,
                "gtol": gtol,
                "eps": eps,
                "maxfun": maxfun,
                "maxiter": maxiter,
                "maxls": maxls,
            },
        )
        # store the results after processing them
        this_result = postprocess_params(*this_result_struct.x)
        for param_idx in range(n_variables):
            param_history[param_idx].append(this_result[param_idx])

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
                    converged[param_idx] = diff < converge_criteria

                # then set the new last value
                param_std_last[param_idx] = this_std

    # then we're done!
    return initial_result, np.array(param_history), success


# ======================================================================================
#
# Then go through the catalog
#
# ======================================================================================
for row in tqdm(clusters_table):
    # create the snapshot. We use ceiling to get the integer pixel values as
    # python indexing does not include the final value.
    x_cen = int(np.ceil(row["x"]))
    y_cen = int(np.ceil(row["y"]))

    # Get the snapshot, based on the size desired.
    # Since we took the ceil of the center, go more in the negative direction (i.e.
    # use ceil to get the minimum values). This only matters if the snapshot size is
    # odd
    x_min = x_cen - int(np.ceil(snapshot_size / 2.0))
    x_max = x_cen + int(np.floor(snapshot_size / 2.0))
    y_min = y_cen - int(np.ceil(snapshot_size / 2.0))
    y_max = y_cen + int(np.floor(snapshot_size / 2.0))

    data_snapshot = image_data[y_min:y_max, x_min:x_max].copy()
    error_snapshot = sigma_data[y_min:y_max, x_min:x_max].copy()
    mask_snapshot = mask_data[y_min:y_max, x_min:x_max].copy()

    mask_snapshot = fit_utils.handle_mask(mask_snapshot, row["ID"])

    # then do this fitting!
    results, history, success = fit_model(data_snapshot, error_snapshot, mask_snapshot)

    # Then add these values to the table
    row["num_boostrapping_iterations"] = len(history[0])
    row["success"] = success

    row["central_surface_brightness_best"] = 10 ** results[0]
    row["x_pix_snapshot_oversampled_best"] = results[1]
    row["y_pix_snapshot_oversampled_best"] = results[2]
    row["x_fitted_best"] = x_min + fit_utils.oversampled_to_image(
        results[1], oversampling_factor
    )
    row["y_fitted_best"] = y_min + fit_utils.oversampled_to_image(
        results[2], oversampling_factor
    )
    row["scale_radius_pixels_best"] = results[3]
    row["axis_ratio_best"] = results[4]
    row["position_angle_best"] = results[5]
    row["power_law_slope_best"] = results[6]
    row["local_background_best"] = results[7]

    row["central_surface_brightness"] = [10 ** v for v in history[0]]
    row["x_pix_snapshot_oversampled"] = history[1]
    row["y_pix_snapshot_oversampled"] = history[2]
    row["x_fitted"] = [
        x_min + fit_utils.oversampled_to_image(v, oversampling_factor)
        for v in history[1]
    ]
    row["y_fitted"] = [
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
