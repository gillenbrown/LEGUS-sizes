"""
fit.py - Do the fitting of the clusters in the image

This takes 4 arguments:
- Path where the resulting cluster catalog will be saved
- Path to the fits image containing the PSF
- Oversampling factor used when creating the PSF
- Path to the sigma image containing uncertainties for each pixel
- Path to the cleaned cluster catalog
"""
from pathlib import Path
import sys

from astropy import table
from astropy import nddata
from astropy.io import fits
import photutils
import betterplotlib as bpl
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
from scipy import signal
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

image_data, _ = utils.get_f555w_drc_image(final_catalog.parent.parent)
psf = fits.open(psf_path)["PRIMARY"].data
# the convolution requires the psf to be normalized, and without any negative values
psf = np.maximum(psf, 0)
psf /= np.sum(psf)

sigma_data = fits.open(sigma_image_path)["PRIMARY"].data
clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")

pixel_scale_arcsec = utils.get_f555w_pixel_scale_arcsec(final_catalog.parent.parent)
pixel_scale_pc = utils.get_f555w_pixel_scale_pc(final_catalog.parent.parent)

# set the size of the images we'll use
snapshot_size = 30
snapshot_size_oversampled = snapshot_size * oversampling_factor

# Also add the new columns to the table, that we will fill as we fit
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


def calculate_chi_squared(params, cluster_snapshot, error_snapshot, x_idxs, y_idxs):
    """
    Calculate the chi-squared value for a given set of parameters.

    :param params: Tuple of parameters of the EFF profile
    :param cluster_snapshot: Cluster snapshot
    :param error_snapshot: Error snapshot
    :param x_idxs: X indices of the pixels to be used when calculating the chi squared.
    :param y_idxs: Y indices of the pixels to be used when calculating the chi squared.
                   This should correspond to the x_idxs, such that the first pixels is
                   at (x_idxs[0], y_idxs[0])
    :return:
    """
    _, _, model_snapshot = create_model_image(*params)
    assert model_snapshot.shape == cluster_snapshot.shape
    assert model_snapshot.shape == error_snapshot.shape

    diffs = cluster_snapshot - model_snapshot
    sigma_snapshot = diffs / error_snapshot
    # then pick the desirec pixels out to be used in the fit
    sum_squared = np.sum(sigma_snapshot[y_idxs, x_idxs] ** 2)
    dof = len(x_idxs) - 8
    return sum_squared / dof


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
    star_finder = photutils.detection.IRAFStarFinder(
        threshold=5 * np.min(uncertainty_snapshot),
        fwhm=2.0,
        exclude_border=False,
        sharphi=5.0,
    )
    peaks_table = star_finder.find_stars(data_snapshot)

    # this will be None if nothing was found
    if peaks_table is None:
        return uncertainty_snapshot

    # then delete any stars near the center
    center = uncertainty_snapshot.shape[0] / 2
    to_remove = []
    for idx in range(len(peaks_table)):
        x = peaks_table[idx]["xcentroid"]
        y = peaks_table[idx]["ycentroid"]
        if distance(center, center, x, y) < 5:
            to_remove.append(idx)
    peaks_table.remove_rows(to_remove)

    # then remove any pixels within 2 FWHM of each source
    mask = np.ones(uncertainty_snapshot.shape)
    for x_idx in range(uncertainty_snapshot.shape[1]):
        x = x_idx + 0.5  # to get pixel center
        for y_idx in range(uncertainty_snapshot.shape[0]):
            y = y_idx + 0.5  # to get pixel center
            # then go through each row and see if it's close
            for row in peaks_table:
                x_cen = row["xcentroid"]
                y_cen = row["ycentroid"]
                radius = 1.5 * row["fwhm"]
                if distance(x, y, x_cen, y_cen) < radius:
                    mask[y_idx][x_idx] = 0.0
    return mask


def create_good_mask_pixels(mask):
    """
    Create the list of good pixel indices that are in the mask

    :param mask: 2D array holding the mask, where 1 values are good, zero is bad.
    :return: Two lists, containing the x and y pixel values of the good pixels in
             this regions. They will be ordered so the values correspond (i.e. the first
             good pixel will be at xs[0], ys[0].
    """
    xs = []
    ys = []
    for x_idx in range(mask.shape[1]):
        for y_idx in range(mask.shape[0]):
            if mask[y_idx][x_idx] == 1:
                xs.append(x_idx)
                ys.append(y_idx)
    # if something is wrong, just return everything
    if len(xs) == 0:
        xs = [int(x) for x in range(mask.shape[1])]
        ys = [int(y) for y in range(mask.shape[0])]
    return np.array(xs), np.array(ys)


def fit_model(data_snapshot, uncertainty_snapshot, mask):
    """
    Fits an EFF model to the data passed in, using bootstrapping.

    :param data_snapshot: 2D array holding the pixel values (in units of electrons)
    :param uncertainty_snapshot: 2D array holding the uncertainty in the pixel values,
                                 in units of electrons.
    :param mask: 2D array holding the mask, where 1 is a good pixel, zero is bad.
    :return: A two-item tuple containing: the fit parameters to all pixels, and the
             history of all parameters took throughout the bootstrapping

    """
    # Create the initial guesses for the parameters
    params = (
        np.log10(np.max(data_snapshot) * 3),  # log of peak brightness.
        # Increase that to account for bins, as peaks will be averaged lower.
        snapshot_size_oversampled / 2.0,  # X center in the oversampled snapshot
        snapshot_size_oversampled / 2.0,  # Y center in the oversampled snapshot
        2 * oversampling_factor,  # scale radius, in oversampled pixels
        0.8,  # axis ratio
        0,  # position angle
        2.0,  # power law slope
        0,  # background
    )

    # set bounds on each parameter
    bounds = (
        # log of peak brightness. The minimum allowed will be the sky value, and the
        # maximum will be 1 order of magnitude above the first guess.
        (np.log10(np.min(uncertainty_snapshot)), params[0] + 1),
        # X and Y center in oversampled pixels
        (0.3 * snapshot_size_oversampled, 0.7 * snapshot_size_oversampled),
        (0.3 * snapshot_size_oversampled, 0.7 * snapshot_size_oversampled),
        (0.1, snapshot_size_oversampled),  # scale radius in oversampled pixels
        (0.01, 1),  # axis ratio
        (-np.pi, np.pi),  # position angle
        (0, None),  # power law slope
        (-np.max(data_snapshot), np.max(data_snapshot)),  # background
    )

    # then get the list of good pixel values that can be used when fitting
    good_xs, good_ys = create_good_mask_pixels(mask)

    # first get the results when all good pixels are used, to be used as a starting
    # point when bootstrapping is done, to save time.
    initial_result = optimize.minimize(
        calculate_chi_squared,
        args=(data_snapshot, uncertainty_snapshot, good_xs, good_ys),
        x0=params,
        bounds=bounds,
    )

    # Then we do bootstrapping
    n_variables = len(initial_result.x)
    param_history = [[] for _ in range(n_variables)]
    param_std_last = [np.inf for _ in range(n_variables)]

    converge_criteria = 0.2
    converged = [False for _ in range(n_variables)]
    check_spacing = 10
    iteration = 0
    while not all(converged):
        iteration += 1
        # sample the xs and ys
        idxs = np.random.randint(0, len(good_xs), len(good_xs))
        this_xs = good_xs[idxs]
        this_ys = good_ys[idxs]

        # fit to this selection of pixels
        this_result = optimize.minimize(
            calculate_chi_squared,
            args=(data_snapshot, uncertainty_snapshot, this_xs, this_ys),
            x0=initial_result.x,
            bounds=bounds,
        )

        # store the results
        for param_idx in range(n_variables):
            param_history[param_idx].append(this_result.x[param_idx])

        # then check if we're converged
        if iteration % check_spacing == 0:
            # go through each one that's not converged yet
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


def plot_model_set(cluster_snapshot, uncertainty_snapshot, mask, params, savename):
    model_image, model_psf_image, model_psf_bin_image = create_model_image(*params)

    diff_image = cluster_snapshot - model_psf_bin_image
    sigma_image = diff_image / uncertainty_snapshot
    sigma_image *= mask

    # Use the data image to get the normalization that will be used in all plots
    vmax = max(np.max(model_psf_bin_image), np.max(cluster_snapshot))
    data_norm = colors.SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=0.01 * vmax)
    sigma_norm = colors.Normalize(vmin=-10, vmax=10)
    # the maximum for the uncertainty should be the maximum that's not infinite. We
    # make it a bit higher to make it clear where the masked regions are
    vmax_u = 1.2 * np.max(uncertainty_snapshot[np.isfinite(uncertainty_snapshot)])
    u_norm = colors.Normalize(0, vmax=vmax_u)
    m_norm = colors.Normalize(0, 1)

    data_cmap = bpl.cm.lisbon
    sigma_cmap = cmocean.cm.curl
    u_cmap = cmocean.cm.deep_r
    m_cmap = cmocean.cm.gray_r

    fig = plt.figure(figsize=[38, 5])
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=7,
        wspace=0.05,
        hspace=0,
        left=0.01,
        right=0.96,
        bottom=0.1,
        top=0.85,
    )
    ax0 = fig.add_subplot(gs[0], projection="bpl")
    ax1 = fig.add_subplot(gs[1], projection="bpl")
    ax2 = fig.add_subplot(gs[2], projection="bpl")
    ax3 = fig.add_subplot(gs[3], projection="bpl")
    ax4 = fig.add_subplot(gs[4], projection="bpl")
    ax5 = fig.add_subplot(gs[5], projection="bpl")
    ax6 = fig.add_subplot(gs[6], projection="bpl")

    ax0.imshow(model_image, norm=data_norm, cmap=data_cmap, origin="lower")
    ax1.imshow(model_psf_bin_image, norm=data_norm, cmap=data_cmap, origin="lower")
    d_im = ax2.imshow(cluster_snapshot, norm=data_norm, cmap=data_cmap, origin="lower")
    s_im = ax3.imshow(sigma_image, norm=sigma_norm, cmap=sigma_cmap, origin="lower")
    u_im = ax4.imshow(uncertainty_snapshot, norm=u_norm, cmap=u_cmap, origin="lower")
    ax5.imshow(mask, norm=m_norm, cmap=m_cmap, origin="lower")

    # the last one just has the list of parameters
    ax6.easy_add_text(
        f"log(peak brightness) = {params[0]:.2f}\n"
        f"x center = {params[1] / oversampling_factor:.2f}\n"
        f"y center = {params[2]/ oversampling_factor:.2f}\n"
        f"scale radius [pixels] = {params[3] / oversampling_factor:.2f}\n"
        f"q (axis ratio) = {params[4]:.2f}\n"
        f"position angle = {params[5]:.2f}\n"
        f"$\eta$ (power law slope) = {params[6]:.2f}\n"
        f"background = {params[7]:.2f}\n",
        "center left",
    )

    fig.colorbar(d_im, ax=ax2)
    fig.colorbar(s_im, ax=ax3)
    fig.colorbar(u_im, ax=ax4)

    ax0.set_title("Raw Model")
    ax1.set_title("Model Convolved\nwith PSF and Binned")
    ax2.set_title("Data")
    ax3.set_title("(Data - Model)/Uncertainty")
    ax4.set_title("Uncertainty")
    ax5.set_title("Mask")

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.remove_labels("both")
        ax.remove_spines(["all"])

    fig.savefig(final_catalog.parent / "cluster_fit_plots" / savename, dpi=100)
    plt.close(fig)


# ======================================================================================
#
# Then go through the catalog
#
# ======================================================================================
for row in tqdm(clusters_table):
    # create the snapshot
    x_cen = int(np.floor(row["x_pix_single"]))
    y_cen = int(np.floor(row["y_pix_single"]))

    # Get the snapshot, based on the size desired. Since we took the floor of the
    # center, go farther in that direction (i.e. use ceil if the number is odd)
    x_min = x_cen - int(np.ceil(snapshot_size / 2.0))
    x_max = x_cen + int(np.floor(snapshot_size / 2.0))
    y_min = y_cen - int(np.ceil(snapshot_size / 2.0))
    y_max = y_cen + int(np.floor(snapshot_size / 2.0))

    data_snapshot = image_data[y_min:y_max, x_min:x_max]
    error_snapshot = sigma_data[y_min:y_max, x_min:x_max]
    # create the mask with the uncertainty image
    mask = mask_image(data_snapshot, error_snapshot)

    results, history = fit_model(data_snapshot, error_snapshot, mask)

    plot_model_set(
        data_snapshot, error_snapshot, mask, results, f"debug_{row['ID']:04}.png"
    )

    # Then add these values to the table
    row["num_boostrapping_iterations"] = len(history[0])

    row["central_surface_brightness_best"] = 10 ** results[0]
    row["x_pix_single_fitted_best"] = x_min + results[1] / oversampling_factor
    row["y_pix_single_fitted_best"] = y_min + results[2] / oversampling_factor
    row["scale_radius_pixels_best"] = results[3] / oversampling_factor
    row["axis_ratio_best"] = results[4]
    row["position_angle_best"] = results[5]
    row["power_law_slope_best"] = results[6]
    row["local_background_best"] = results[7]

    row["central_surface_brightness"] = [10 ** v for v in history[0]]
    row["x_pix_single_fitted"] = [x_min + v / oversampling_factor for v in history[1]]
    row["y_pix_single_fitted"] = [y_min + v / oversampling_factor for v in history[2]]
    row["scale_radius_pixels"] = [v / oversampling_factor for v in history[3]]
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

clusters_table.write(str(final_catalog), format="hdf5", overwrite=True)
