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
from astropy import convolution
from astropy.io import fits
import betterplotlib as bpl
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
import cmocean
from tqdm import tqdm

import time

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
sigma_data = fits.open(sigma_image_path)["PRIMARY"].data
clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")

pixel_scale_arcsec = utils.get_f555w_pixel_scale_arcsec(final_catalog.parent.parent)
pixel_scale_pc = utils.get_f555w_pixel_scale_pc(final_catalog.parent.parent)

# Also add the new columns to the table, that we will fill as we fit
clusters_table["x_pix_single_fitted"] = -99.9
clusters_table["y_pix_single_fitted"] = -99.9
clusters_table["central_surface_brightness"] = -99.9
clusters_table["scale_radius_pixels"] = -99.9
clusters_table["scale_radius_arcseconds"] = -99.9
clusters_table["scale_radius_pc"] = -99.9
clusters_table["axis_ratio"] = -99.9
clusters_table["position_angle"] = -99.9
clusters_table["power_law_slope"] = -99.9
clusters_table["local_background"] = -99.9

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
    # using boundary wrap stops edge effects from reducing the values near the boundary
    return convolution.convolve_fft(in_array, psf, boundary="wrap")


def bin_data_2d(data):
    """ Bin a 2d array into square bins determined by the oversampling factor """
    x_length_new = data.shape[0] // oversampling_factor
    y_length_new = data.shape[1] // oversampling_factor

    binned_data = np.zeros((x_length_new, y_length_new))

    for x_new in range(x_length_new):
        for y_new in range(y_length_new):
            x_old_min = oversampling_factor * x_new
            x_old_max = oversampling_factor * (x_new + 1)
            y_old_min = oversampling_factor * y_new
            y_old_max = oversampling_factor * (y_new + 1)

            data_subset = data[x_old_min:x_old_max, y_old_min:y_old_max]
            binned_data[x_new, y_new] = np.mean(data_subset)

    return binned_data


def create_model_image(log_mu_0, x_c, y_c, a, q, theta, eta, background):
    """ Create a model image using the EFF parameters. """
    # first generate the x and y pixel coordinates of the model image, which will
    # be the same size as the PSF
    x_values = np.zeros(psf.shape)
    y_values = np.zeros(psf.shape)

    for x in range(psf.shape[1]):
        x_values[:, x] = x
    for y in range(psf.shape[0]):
        y_values[y, :] = y

    model_image = eff_profile_2d(
        x_values, y_values, log_mu_0, x_c, y_c, a, q, theta, eta
    )
    model_image += background
    model_psf_image = convolve_with_psf(model_image)
    model_psf_bin_image = bin_data_2d(model_psf_image)

    # return all of these, since we'll want to use them when plotting
    return model_image, model_psf_image, model_psf_bin_image


def calculate_chi_squared(params, cluster_snapshot, error_snapshot):
    """
    Calculate the chi-squared value for a given set of parameters.

    :param params: Tuple of parameters of the EFF profile
    :param cluster_snapshot: Cluster snapshot
    :param error_snapshot:
    :return:
    """
    _, _, model_snapshot = create_model_image(*params)
    assert model_snapshot.shape == cluster_snapshot.shape
    assert model_snapshot.shape == error_snapshot.shape

    diffs = cluster_snapshot - model_snapshot
    sigma_snapshot = diffs / error_snapshot
    sum_squared = np.sum(sigma_snapshot ** 2)
    dof = model_snapshot.size - 8
    return sum_squared - dof


def fit_model(data_snapshot, uncertainty_snapshot):
    """
    Fits an EFF model to the data passed in.

    :param data_snapshot: 2D array holding the pixel values (in units of electrons)
    :param uncertainty_snapshot: 2D array holding the uncertainty in the pixel values,
                                 in units of electrons.
    :return: List of fitted parameters
    """
    # Create the initial guesses for the parameters
    params = (
        np.log10(np.max(data_snapshot) * 3),  # log of peak brightness.
        # Increase that to account for bins, as peaks will be averaged lower.
        psf.shape[1] / 2.0,  # X center in the oversampled snapshot
        psf.shape[0] / 2.0,  # Y center in the oversampled snapshot
        2 * oversampling_factor,  # scale radius, in oversampled pixels
        0.9,  # axis ratio
        0,  # position angle
        2.0,  # power law slope
        0,  # background
    )

    # set bounds on each parameter
    bounds = (
        # log of peak brightness. The minimum allowed will be the sky value, and the
        # maximum will be 1 order of magnitude above the first guess.
        (np.log10(np.min(uncertainty_snapshot)), params[0] + 1),
        (0.3 * psf.shape[1], 0.7 * psf.shape[1]),  # X center
        (0.3 * psf.shape[0], 0.7 * psf.shape[0]),  # Y center
        (1, psf.shape[0]),  # scale radius in oversampled pixels
        (0.1, 1),  # axis ratio
        (-np.pi, np.pi),  # position angle
        (1.0001, 5),  # power law slope
        (-np.max(data_snapshot), np.max(data_snapshot)),  # background
    )

    results = optimize.minimize(
        calculate_chi_squared,
        args=(data_snapshot, error_snapshot),
        x0=params,
        bounds=bounds,
    )
    return results.x


def plot_model_set(cluster_snapshot, uncertainty_snapshot, params, savename):
    model_image, model_psf_image, model_psf_bin_image = create_model_image(*params)

    diff_image = cluster_snapshot - model_psf_bin_image
    sigma_image = diff_image / uncertainty_snapshot

    # Use the data image to get the normalization that will be used in all plots
    vmax = max(np.max(model_image), np.max(cluster_snapshot))
    data_norm = colors.SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=0.01 * vmax)
    sigma_norm = colors.Normalize(vmin=-10, vmax=10)
    u_norm = colors.Normalize(0, vmax=np.max(uncertainty_snapshot))

    data_cmap = bpl.cm.lisbon
    sigma_cmap = cmocean.cm.curl
    u_cmap = cmocean.cm.deep_r

    fig = plt.figure(figsize=[38, 5])
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=7,
        width_ratios=[10, 10, 10, 10, 10, 10, 10],
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
    # cax1 = fig.add_subplot(gs[5], projection="bpl")
    # cax2 = fig.add_subplot(gs[7], projection="bpl")
    # cax3 = fig.add_subplot(gs[9], projection="bpl")

    ax0.imshow(model_image, norm=data_norm, cmap=data_cmap, origin="lower")
    ax1.imshow(model_psf_image, norm=data_norm, cmap=data_cmap, origin="lower")
    ax2.imshow(model_psf_bin_image, norm=data_norm, cmap=data_cmap, origin="lower")
    ax3.imshow(cluster_snapshot, norm=data_norm, cmap=data_cmap, origin="lower")
    d_im = ax4.imshow(diff_image, norm=data_norm, cmap=data_cmap, origin="lower")
    s_im = ax5.imshow(sigma_image, norm=sigma_norm, cmap=sigma_cmap, origin="lower")
    u_im = ax6.imshow(uncertainty_snapshot, norm=u_norm, cmap=u_cmap, origin="lower")

    cbar_d = fig.colorbar(d_im, ax=ax4)
    cbar_s = fig.colorbar(s_im, ax=ax5)
    cbar_u = fig.colorbar(u_im, ax=ax6)

    # cbar_d.set_label("Pixel Values [Electrons]")
    # cbar_s.set_label("Error [Standard Deviations]")
    # cbar_u.set_label("Pixel Uncertainty [Electrons]")

    ax0.set_title("Raw Model")
    ax1.set_title("Model Convolved\nwith PSF")
    ax2.set_title("Model Convolved\nwith PSF and Binned")
    ax3.set_title("Data")
    ax4.set_title("Data - Model")
    ax5.set_title("(Data - Model)/Uncertainty")
    ax6.set_title("Uncertainty")

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.remove_labels("both")
        ax.remove_spines(["all"])

    fig.savefig(final_catalog.parent / "cluster_fit_plots" / savename, dpi=100)


# ======================================================================================
#
# Then go through the catalog
#
# ======================================================================================
for row in tqdm(clusters_table):
    # create the snapshot
    x_cen = int(np.floor(row["x_pix_single"]))
    y_cen = int(np.floor(row["y_pix_single"]))

    # We want a 31 pixel wide shapshot. Since we do the floor we go 16 to the left, then
    # also go 16 to the right, but that will be cut down to 15 since the last index
    # won't be included
    x_min = x_cen - 16
    x_max = x_cen + 15
    y_min = y_cen - 16
    y_max = y_cen + 15

    data_snapshot = image_data[y_min:y_max, x_min:x_max]
    error_snapshot = sigma_data[y_min:y_max, x_min:x_max]

    results = fit_model(data_snapshot, error_snapshot)

    plot_model_set(data_snapshot, error_snapshot, results, f"debug_{row['ID']:04}.png")

    # Then add these values to the table
    row["central_surface_brightness"] = 10 ** results[0]
    row["x_pix_single_fitted"] = x_min + results[1] / oversampling_factor
    row["y_pix_single_fitted"] = y_min + results[2] / oversampling_factor
    row["scale_radius_pixels"] = results[3] / oversampling_factor
    row["scale_radius_arcseconds"] = row["scale_radius_pixels"] * pixel_scale_arcsec
    row["scale_radius_pc"] = row["scale_radius_pixels"] * pixel_scale_pc
    row["axis_ratio"] = results[4]
    row["position_angle"] = results[5]
    row["power_law_slope"] = results[6]
    row["local_background"] = results[7]

# ======================================================================================
#
# Then write this output catalog
#
# ======================================================================================
clusters_table.write(final_catalog, format="ascii.ecsv")
