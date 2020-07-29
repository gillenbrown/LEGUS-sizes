"""
fit.py - Do the fitting of the clusters in the image

This takes 6 or 7 arguments:
- Path where the resulting cluster catalog will be saved
- Path to the fits image containing the PSF
- Oversampling factor used when creating the PSF
- Path to the sigma image containing uncertainties for each pixel
- Path to the cleaned cluster catalog
- Size of the fitting region to be used
- Optional argument that must be "ryon_like" if present. If it is present, masking will
  not be done, and the power law slope will be restricted to be greater than 1.
"""
from pathlib import Path
import sys
from collections import defaultdict

from astropy import table, nddata, stats
from astropy.io import fits
import photutils
import betterplotlib as bpl
from matplotlib import colors, gridspec
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize, signal, special
import cmocean
from tqdm import tqdm

import utils

bpl.set_style()

# ======================================================================================
#
# Get the parameters the user passed in, load images and catalogs
#
# ======================================================================================
final_catalog = Path(sys.argv[1]).absolute()
psf_path = Path(sys.argv[2]).absolute()
oversampling_factor = int(sys.argv[3])
sigma_image_path = Path(sys.argv[4]).absolute()
cluster_catalog_path = Path(sys.argv[5]).absolute()
snapshot_size = int(sys.argv[6])
if len(sys.argv) > 7:
    if len(sys.argv) != 8 or sys.argv[7] != "ryon_like":
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
clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")

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
# Create the functions that will be used in the fitting procedure
#
# ======================================================================================
def eff_profile_2d(x, y, log_mu_0, x_c, y_c, a, q, theta, eta):
    """
    2-dimensional EFF profile, in pixel units

    :param x: X pixel values
    :param y: Y pixel values
    :param log_mu_0: Log of the central surface brightness
    :param x_c: Center X pixel coordinate
    :param y_c: Center Y pixel coordinate
    :param a: Scale radius in the major axis direction
    :param q: Axis ratio. The small axis will have scale length qa
    :param theta: Position angle
    :param eta: Power law slope
    :return: Values of the function at the x,y coordinates passed in
    """
    x_prime = (x - x_c) * np.cos(theta) + (y - y_c) * np.sin(theta)
    y_prime = -(x - x_c) * np.sin(theta) + (y - y_c) * np.cos(theta)

    x_term = (x_prime / a) ** 2
    y_term = (y_prime / (q * a)) ** 2
    return (10 ** log_mu_0) * (1 + x_term + y_term) ** (-eta)


def convolve_with_psf(in_array):
    """ Convolve an array with the PSF """
    # Scipy FFT based convolution was tested to be the fastest. It does have edge
    # affects, but those are minimized by using our padding. We do have to modify the
    # psf to have
    return signal.fftconvolve(in_array, psf, mode="same")


def bin_data_2d(data):
    """ Bin a 2d array into square bins determined by the oversampling factor """
    # Astropy has a convenient function to do this
    bin_factors = [oversampling_factor, oversampling_factor]
    return nddata.block_reduce(data, bin_factors, np.mean)


def create_model_image(log_mu_0, x_c, y_c, a, q, theta, eta, background):
    """ Create a model image using the EFF parameters. """
    # a is passed in in regular coordinates, shift it to the oversampled ones
    a *= oversampling_factor

    # first generate the x and y pixel coordinates of the model image. We will have
    # an array that's the same size as the cluster snapshot in oversampled pixels,
    # plus padding to account for zero-padded boundaries in the FFT convolution
    padding = 5 * oversampling_factor  # 5 regular pixels on each side
    box_length = snapshot_size_oversampled + 2 * padding

    # correct the center to be at the center of this new array
    x_c_internal = x_c + padding
    y_c_internal = y_c + padding

    x_values = np.zeros([box_length, box_length])
    y_values = np.zeros([box_length, box_length])

    for x in range(box_length):
        x_values[:, x] = x
    for y in range(box_length):
        y_values[y, :] = y

    model_image = eff_profile_2d(
        x_values, y_values, log_mu_0, x_c_internal, y_c_internal, a, q, theta, eta
    )
    # convolve without the background first, to do an ever better job avoiding edge
    # effects, as the model should be zero near the boundaries anyway, matching the zero
    # padding scipy does.
    model_psf_image = convolve_with_psf(model_image)
    model_image += background
    model_psf_image += background

    # crop out the padding before binning the data
    model_image = model_image[padding:-padding, padding:-padding]
    model_psf_image = model_psf_image[padding:-padding, padding:-padding]
    model_psf_bin_image = bin_data_2d(model_psf_image)

    # return all of these, since we'll want to use them when plotting
    return model_image, model_psf_image, model_psf_bin_image


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
    _, _, model_snapshot = create_model_image(*params)
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
    # x and y center have a Gaussian with width of 2 regular pixels, centered on
    # the center of the snapshot
    prior *= gaussian(x_c, snapshot_size_oversampled / 2.0, 3 * oversampling_factor)
    prior *= gaussian(y_c, snapshot_size_oversampled / 2.0, 3 * oversampling_factor)
    prior *= gamma(a, 1.05, 20)
    prior *= gamma(eta - 0.6, 1.05, 20)
    prior *= trapezoid(q, 0.3)
    # have a minimum allowed value, to stop it from being zero if several of these
    # parameters are bad.
    return np.maximum(prior, 1e-50)


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


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def mask_image(data_snapshot, uncertainty_snapshot):
    """
    Create mask image

    This wil use IRAFStarFinder from photutils to find stars in the snapshot. Anything
    more than 5 pixels from the center will be masked, with a radius of FWHM. This
    makes the full width masked twice the FWHM.

    :param data_snapshot: Snapshot to be used to identify sources
    :param uncertainty_snapshot: Snapshow showing the uncertainty.
    :return: masked image, where values with 1 are good, zero is bad.
    """
    mask = np.ones(data_snapshot.shape)
    if ryon_like:  # Ryon did no masking
        return mask

    threshold = 5 * np.min(uncertainty_snapshot)
    star_finder = photutils.detection.IRAFStarFinder(
        threshold=threshold + np.min(data_snapshot),
        fwhm=2.0,  # slightly larger than real PSF, to get extended sources
        exclude_border=False,
        sharplo=0.8,
        sharphi=5,
        roundlo=0.0,
        roundhi=0.5,
        minsep_fwhm=1.0,
    )
    peaks_table = star_finder.find_stars(data_snapshot)

    # this will be None if nothing was found
    if peaks_table is None:
        return mask

    # then delete any stars near the center
    center = data_snapshot.shape[0] / 2
    to_remove = []
    for idx in range(len(peaks_table)):
        row = peaks_table[idx]
        x = row["xcentroid"]
        y = row["ycentroid"]
        if (
            distance(center, center, x, y) < 6
            or row["peak"] < row["sky"]
            # peak is sky-subtracted. This ^ removes ones that aren't very far above
            # a high sky background. This cut stops substructure in clusters from
            # being masked.
            or row["peak"] < threshold
        ):
            to_remove.append(idx)
    peaks_table.remove_rows(to_remove)

    # we may have gotten rid of all the peaks, if so return
    if len(peaks_table) == 0:
        return mask

    # then remove any pixels within 2 FWHM of each source
    for x_idx in range(data_snapshot.shape[1]):
        x = x_idx + 0.5  # to get pixel center
        for y_idx in range(data_snapshot.shape[0]):
            y = y_idx + 0.5  # to get pixel center
            # then go through each row and see if it's close
            for row in peaks_table:
                x_cen = row["xcentroid"]
                y_cen = row["ycentroid"]
                radius = 2.0 * row["fwhm"]
                if distance(x, y, x_cen, y_cen) < radius:
                    mask[y_idx][x_idx] = 0.0
    return mask


def create_plot_name(id, bootstrapping_iteration=None):
    name = f"{galaxy_name}_{id:04}_size_{snapshot_size}_"

    if ryon_like:
        name += "ryon_like_"
    else:
        name += "final_"

    if bootstrapping_iteration is None:
        name += "best_fit"
    else:
        name += f"bootstrapping_i{bootstrapping_iteration:04}"

    return name + ".png"


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
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            if original_mask[y, x] == 1:  # only keep pixels not already masked out
                if distance(x, y, x_c, y_c) <= 9:
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


def fit_model(data_snapshot, uncertainty_snapshot, mask, id_num):
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
    bounds = [
        # log of peak brightness. The minimum allowed will be the sky value, and the
        # maximum will be 2 orders of magnitude above the first guess.
        (np.log10(np.min(uncertainty_snapshot)), params[0] + 2),
        (0, snapshot_size_oversampled),  # X center
        (0, snapshot_size_oversampled),  # Y center
        (1e-10, None),  # scale radius in regular pixels.
        (1e-10, 1),  # axis ratio
        (0, np.pi),  # position angle
        (0.6, None),  # power law slope
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

    # plot this initial result
    plot_model_set(
        data_snapshot,
        uncertainty_snapshot,
        mask,
        initial_result.x,
        create_plot_name(id_num),
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

            # # plot this iteration
            # plot_model_set(
            #     data_snapshot,
            #     uncertainty_snapshot,
            #     temp_mask,
            #     this_result.x,
            #     create_plot_name(id_num, iteration),
            # )

    # then we're done!
    return initial_result.x, np.array(param_history)


def oversampled_to_image(x):
    """
    Turn oversampled pixel coordinates into regular pixel coordinates.

    There are two affects here: first the pixel size is different, and the centers
    (where the pixel is defined to be) are not aligned.

    :param x: Location in oversampled pixel coordinates
    :return: Location in regular pixel coordinates
    """
    # first have to correct for the pixel size
    x /= oversampling_factor
    # then the oversampled pixel centers are offset from the regular pixels,
    # so we need to correct for that too
    if oversampling_factor == 2:
        return x - 0.25
    else:
        raise ValueError("Think about this more")


def plot_model_set(cluster_snapshot, uncertainty_snapshot, mask, params, savename):
    model_image, model_psf_image, model_psf_bin_image = create_model_image(*params)

    diff_image = cluster_snapshot - model_psf_bin_image
    sigma_image = diff_image / uncertainty_snapshot
    # have zeros in the sigma image where the mask has zeros, but leave it unmodified
    # otherwise
    sigma_image *= np.minimum(mask, 1.0)

    # set up the normalizations and colormaps
    # Use the data image to get the normalization that will be used in all plots. Base
    # it on the data so that it is the same in all bootstrap iterations
    vmax = 2 * np.max(cluster_snapshot)
    linthresh = 3 * np.min(uncertainty_snapshot)
    data_norm = colors.SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, base=10)
    sigma_norm = colors.Normalize(vmin=-10, vmax=10)
    u_norm = colors.Normalize(0, vmax=1.2 * np.max(uncertainty_snapshot))
    m_norm = colors.Normalize(0, vmax=np.max(mask))

    data_cmap = bpl.cm.lisbon
    sigma_cmap = cmocean.cm.tarn  # "bwr_r" also works
    u_cmap = cmocean.cm.deep_r
    m_cmap = cmocean.cm.gray_r

    # create the figure and add all the subplots
    fig = plt.figure(figsize=[20, 15])
    gs = gridspec.GridSpec(
        nrows=6,
        ncols=4,
        width_ratios=[10, 10, 1, 15],  # have a dummy spacer column
        wspace=0.1,
        hspace=0.7,
        left=0.01,
        right=0.98,
        bottom=0.06,
        top=0.94,
    )
    ax_r = fig.add_subplot(gs[0:2, 0], projection="bpl")  # raw model
    ax_f = fig.add_subplot(gs[0:2, 1], projection="bpl")  # full model (f for fit)
    ax_d = fig.add_subplot(gs[2:4, 1], projection="bpl")  # data
    ax_s = fig.add_subplot(gs[2:4, 0], projection="bpl")  # sigma difference
    ax_u = fig.add_subplot(gs[4:, 1], projection="bpl")  # uncertainty
    ax_m = fig.add_subplot(gs[4:, 0], projection="bpl")  # mask
    ax_pd = fig.add_subplot(gs[0:3, 3], projection="bpl")  # radial profile differential
    ax_pc = fig.add_subplot(gs[3:, 3], projection="bpl")  # radial profile cumulative

    # show the images in their respective panels
    common_data = {"norm": data_norm, "cmap": data_cmap}
    r_im = ax_r.imshow(model_image, **common_data, origin="lower")
    f_im = ax_f.imshow(model_psf_bin_image, **common_data, origin="lower")
    d_im = ax_d.imshow(cluster_snapshot, **common_data, origin="lower")
    s_im = ax_s.imshow(sigma_image, norm=sigma_norm, cmap=sigma_cmap, origin="lower")
    u_im = ax_u.imshow(uncertainty_snapshot, norm=u_norm, cmap=u_cmap, origin="lower")
    m_im = ax_m.imshow(mask, norm=m_norm, cmap=m_cmap, origin="lower")

    fig.colorbar(r_im, ax=ax_r)
    fig.colorbar(f_im, ax=ax_f)
    fig.colorbar(d_im, ax=ax_d)
    fig.colorbar(s_im, ax=ax_s)
    fig.colorbar(u_im, ax=ax_u)
    fig.colorbar(m_im, ax=ax_m)

    ax_r.set_title("Raw Cluster Model")
    ax_f.set_title("Model Convolved\nwith PSF and Binned")
    ax_d.set_title("Data")
    ax_s.set_title("(Data - Model)/Uncertainty")
    ax_u.set_title("Uncertainty")
    ax_m.set_title("Mask")

    for ax in [ax_r, ax_f, ax_d, ax_s, ax_u, ax_m]:
        ax.remove_labels("both")
        ax.remove_spines(["all"])

    # Then make the radial plots. first background subtract
    cluster_snapshot -= params[7]
    model_image -= params[7]
    model_psf_image -= params[7]
    model_psf_bin_image -= params[7]

    c_d = bpl.color_cycle[0]
    c_m = bpl.color_cycle[1]
    # the center is in oversampled coords, fix that
    x_c = oversampled_to_image(params[1])
    y_c = oversampled_to_image(params[2])
    # When fitting I treated the center of the pixels as the integer location, so do
    # that here too
    radii, model_ys, data_ys = [], [], []
    for x in range(model_psf_bin_image.shape[1]):
        for y in range(model_psf_bin_image.shape[0]):
            if mask[y][x] > 0:
                radii.append(distance(x, y, x_c, y_c))
                model_ys.append(model_psf_bin_image[y, x])
                data_ys.append(cluster_snapshot[y, x])

    ax_pd.scatter(radii, data_ys, c=c_d, s=5, alpha=1.0, label="Data")
    ax_pd.scatter(radii, model_ys, c=c_m, s=5, alpha=1.0, label="Model")
    ax_pd.axhline(0, ls=":", c=bpl.almost_black)

    # convert to numpy array to have nice indexing
    radii = np.array(radii)
    model_ys = np.array(model_ys)
    data_ys = np.array(data_ys)

    # then bin this data to make the binned plot
    bin_size = 1.0
    binned_radii, binned_model_ys, binned_data_ys = [], [], []
    for r_min in np.arange(0, int(np.ceil(max(radii))), bin_size):
        r_max = r_min + bin_size
        idx_above = np.where(r_min < radii)
        idx_below = np.where(r_max > radii)
        idx_good = np.intersect1d(idx_above, idx_below)

        if len(idx_good) > 0:
            binned_radii.append(r_min + 0.5 * bin_size)
            binned_model_ys.append(np.mean(model_ys[idx_good]))
            binned_data_ys.append(np.mean(data_ys[idx_good]))

    ax_pd.plot(binned_radii, binned_data_ys, c=c_d, lw=5, label="Binned Data")
    ax_pd.plot(binned_radii, binned_model_ys, c=c_m, lw=5, label="Binned Model")

    ax_pd.legend(loc="upper right")
    ax_pd.add_labels("Radius (pixels)", "Background Subtracted Pixel Value [$e^{-}$]")
    # set min and max values so it's easier to flip through bootstrapping plots
    y_min = np.min(data_snapshot)
    y_max = np.max(data_snapshot)
    # give them a bit of padding
    diff = y_max - y_min
    y_min -= 0.1 * diff
    y_max += 0.1 * diff
    ax_pd.set_limits(0, np.ceil(max(radii)), y_min, y_max)

    # then make the cumulative one. get the radii in order
    idxs_sort = np.argsort(radii)
    model_ys_cumulative = np.cumsum(model_ys[idxs_sort])
    data_ys_cumulative = np.cumsum(data_ys[idxs_sort])

    ax_pc.plot(
        radii[idxs_sort], data_ys_cumulative, c=c_d, label="Data",
    )
    ax_pc.plot(
        radii[idxs_sort], model_ys_cumulative, c=c_m, label="Model",
    )
    ax_pc.set_limits(0, np.ceil(max(radii)), 0, 1.2 * np.max(data_ys_cumulative))
    ax_pc.legend(loc="upper left")
    ax_pc.add_labels(
        "Radius (pixels)", "Cumulative Background Subtracted Pixel Values [$e^{-}$]"
    )

    # the last one just has the list of parameters
    ax_pc.easy_add_text(
        f"log(peak brightness) = {params[0]:.2f}\n"
        f"x center = {oversampled_to_image(params[1]):.2f}\n"
        f"y center = {oversampled_to_image(params[2]):.2f}\n"
        f"scale radius [pixels] = {params[3]:.2f}\n"
        f"q (axis ratio) = {params[4]:.2f}\n"
        f"position angle = {params[5]:.2f}\n"
        f"$\eta$ (power law slope) = {params[6]:.2f}\n"
        f"background = {params[7]:.2f}\n",
        "lower right",
        fontsize=15,
    )

    fig.savefig(final_catalog.parent / "cluster_fit_plots" / savename, dpi=100)
    plt.close(fig)


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

    # Get the snapshot, based on the size desired. I start out creating snapshots a
    # bit larger than desired, so that we can select stars on the borders of the image.
    # Since we took the ceil of the center, go more in the negative direction (i.e.
    # use ceil to get the minimum values). This only matters if the snapshot size is odd
    buffer = 10
    x_min = x_cen - int(np.ceil(snapshot_size / 2.0) + buffer)
    x_max = x_cen + int(np.floor(snapshot_size / 2.0) + buffer)
    y_min = y_cen - int(np.ceil(snapshot_size / 2.0) + buffer)
    y_max = y_cen + int(np.floor(snapshot_size / 2.0) + buffer)

    data_snapshot = image_data[y_min:y_max, x_min:x_max]
    error_snapshot = sigma_data[y_min:y_max, x_min:x_max]
    # create the mask
    mask = mask_image(data_snapshot, error_snapshot)

    # then crop the images back to the desired size
    data_snapshot = data_snapshot[buffer:-buffer, buffer:-buffer]
    error_snapshot = error_snapshot[buffer:-buffer, buffer:-buffer]
    mask = mask[buffer:-buffer, buffer:-buffer]

    # then do this fitting!
    results, history = fit_model(data_snapshot, error_snapshot, mask, row["ID"])

    # Then add these values to the table
    row["num_boostrapping_iterations"] = len(history[0])

    row["central_surface_brightness_best"] = 10 ** results[0]
    row["x_pix_single_fitted_best"] = x_min + oversampled_to_image(results[1])
    row["y_pix_single_fitted_best"] = y_min + oversampled_to_image(results[2])
    row["scale_radius_pixels_best"] = results[3]
    row["axis_ratio_best"] = results[4]
    row["position_angle_best"] = results[5]
    row["power_law_slope_best"] = results[6]
    row["local_background_best"] = results[7]

    row["central_surface_brightness"] = [10 ** v for v in history[0]]
    row["x_pix_single_fitted"] = [x_min + oversampled_to_image(v) for v in history[1]]
    row["y_pix_single_fitted"] = [y_min + oversampled_to_image(v) for v in history[2]]
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
