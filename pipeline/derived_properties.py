"""
The catalogs from the fits just have the fit parameters added. I can then parse
those into various derived properties (mainly effective radius) separately so any
changes in this doesn't require re-running all the fits.

This takes two comand line arguments
1 - The path where the final catalog will be saved
2 - The path of the raw fits catalog
"""

import sys
from pathlib import Path

from astropy import table
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from matplotlib import colors, gridspec
from matplotlib import pyplot as plt
import cmocean
import betterplotlib as bpl

import utils
import fit_utils

bpl.set_style()

# ======================================================================================
#
# Get the parameters the user passed in, load the catalog
#
# ======================================================================================
final_catalog_path = Path(sys.argv[1]).resolve()
fits_catalog_path = Path(sys.argv[2]).resolve()
psf_path = Path(sys.argv[3]).resolve()
oversampling_factor = int(sys.argv[4])
sigma_image_path = Path(sys.argv[5]).absolute()
mask_image_path = Path(sys.argv[6]).absolute()
snapshot_size = int(sys.argv[7])

size_dir = final_catalog_path.parent
home_dir = size_dir.parent
galaxy_name = home_dir.name
image_data, _, _ = utils.get_drc_image(home_dir)
fits_catalog = table.Table.read(fits_catalog_path, format="hdf5")
sigma_data = fits.open(sigma_image_path)["PRIMARY"].data
mask_data = fits.open(mask_image_path)["PRIMARY"].data
psf = fits.open(psf_path)["PRIMARY"].data
# the convolution requires the psf to be normalized, and without any negative values
psf = np.maximum(psf, 0)
psf /= np.sum(psf)

snapshot_size_oversampled = snapshot_size * oversampling_factor

# ======================================================================================
#
# Handle the final output arrays
#
# ======================================================================================
# We want to get the values out of the padded arrays where there are nans
def unpad(padded_array):
    return padded_array[np.where(~np.isnan(padded_array))]


def pad(array, total_length):
    final_array = np.zeros(total_length) * np.nan
    final_array[: len(array)] = array
    return final_array


dist_cols = [
    "x_pix_single_fitted",
    "y_pix_single_fitted",
    "x_pix_snapshot_oversampled",
    "y_pix_snapshot_oversampled",
    "central_surface_brightness",
    "scale_radius_pixels",
    "axis_ratio",
    "position_angle",
    "power_law_slope",
    "local_background",
]
for col in dist_cols:
    fits_catalog[col] = [unpad(row[col]) for row in fits_catalog]

# ======================================================================================
#
# Calculate the errors of the ones with distributions
#
# ======================================================================================
for col in dist_cols:
    fits_catalog[col + "_e+"] = -99.9
    fits_catalog[col + "_e-"] = -99.9
    for row in fits_catalog:
        low, hi = np.percentile(row[col], [15.85, 84.15])
        med = row[col + "_best"]
        row[col + "_e+"] = hi - med
        row[col + "_e-"] = med - low

# ======================================================================================
#
# Define some functions defining various quantities
#
# ======================================================================================
def logistic(eta):
    """
    This is the fit to the slopes as a function of eta

    These slopes are used in the ellipticity correction.
    :param eta: Eta (power law slope)
    :return: The slope to go in ellipticity_correction
    """
    ymax = 0.57902801
    scale = 0.2664717
    eta_0 = 0.92404378
    offset = 0.07298404
    return ymax / (1 + np.exp((eta_0 - eta) / scale)) - offset


def ellipticy_correction(q, eta):
    """
    Correction for ellipticity. This given R_eff(q) / R_eff(q=1)

    This is a generalized form of the simplified form used in Ryon's analysis. It's
    simply a line of arbitrary slope passing through (q=1, correction=1) as circular
    clusters need no correction. This lets us write the correction in point slope form
    as:
    y - 1 = m (q - 1)
    y = 1 + m (q - 1)

    Note that when m = 0.5, this simplifies to y = (1 + q) * 0.5, as used in Ryon.
    The slope here (m) is determined by fitting it as a function of eta.
    """
    return 1 + logistic(eta) * (q - 1)


def eff_profile_r_eff_with_rmax(eta, a, q, rmax):
    """
    Calculate the effective radius of an EFF profile, assuming a maximum radius.

    :param eta: Power law slope of the EFF profile
    :param a: Scale radius of the EFF profile, in any units.
    :param q: Axis ratio of the profile
    :param rmax: Maximum radius for the profile, in the same units as a.
    :return: Effective radius, in the same units as a and rmax
    """
    # This is such an ugly formula, put it in a few steps
    term_1 = 1 + (1 + (rmax / a) ** 2) ** (1 - eta)
    term_2 = (0.5 * (term_1)) ** (1 / (1 - eta)) - 1
    return ellipticy_correction(q, eta) * a * np.sqrt(term_2)


# ======================================================================================
#
# Then add these quantities to the table
#
# ======================================================================================
# add the columns we want to the table
fits_catalog["r_eff_pixels_rmax_15pix_e+"] = -99.9
fits_catalog["r_eff_pixels_rmax_15pix_e-"] = -99.9
fits_catalog["r_eff_pc_rmax_15pix_best"] = -99.9
fits_catalog["r_eff_pc_rmax_15pix_e+"] = -99.9
fits_catalog["r_eff_pc_rmax_15pix_e-"] = -99.9
fits_catalog["r_eff_pc_rmax_15pix_e+_with_dist"] = -99.9
fits_catalog["r_eff_pc_rmax_15pix_e-_with_dist"] = -99.9

# first calculate the best fit value
fits_catalog["r_eff_pixels_rmax_15pix_best"] = eff_profile_r_eff_with_rmax(
    fits_catalog["power_law_slope_best"],
    fits_catalog["scale_radius_pixels_best"],
    fits_catalog["axis_ratio_best"],
    15,  # size of the box
)

# calculate the distribution of all effective radii in each case
for row in fits_catalog:
    # Calculate the effective radius in pixels for all bootstrapping iterations
    all_r_eff_pixels = eff_profile_r_eff_with_rmax(
        row["power_law_slope"],
        row["scale_radius_pixels"],
        row["axis_ratio"],
        15,  # size of the box
    )
    # then get the 1 sigma error range of that
    lo, hi = np.percentile(all_r_eff_pixels, [15.85, 84.15])
    # subtract the middle to get the error range. If the best fit it outside the error
    # range, make the error in that direction zero.
    row["r_eff_pixels_rmax_15pix_e+"] = max(hi - row["r_eff_pixels_rmax_15pix_best"], 0)
    row["r_eff_pixels_rmax_15pix_e-"] = max(row["r_eff_pixels_rmax_15pix_best"] - lo, 0)

    # Then we can convert to pc. First do it without including distance errors
    best, low_e, high_e = utils.pixels_to_pc_with_errors(
        final_catalog_path.parent.parent,
        row["r_eff_pixels_rmax_15pix_best"],
        row["r_eff_pixels_rmax_15pix_e-"],
        row["r_eff_pixels_rmax_15pix_e+"],
        include_distance_err=False,
    )

    row["r_eff_pc_rmax_15pix_best"] = best
    row["r_eff_pc_rmax_15pix_e+"] = high_e
    row["r_eff_pc_rmax_15pix_e-"] = low_e

    # Then recalculate the errors including the distance errors
    _, low_e, high_e = utils.pixels_to_pc_with_errors(
        final_catalog_path.parent.parent,
        row["r_eff_pixels_rmax_15pix_best"],
        row["r_eff_pixels_rmax_15pix_e-"],
        row["r_eff_pixels_rmax_15pix_e+"],
        include_distance_err=True,
    )

    row["r_eff_pc_rmax_15pix_e+_with_dist"] = high_e
    row["r_eff_pc_rmax_15pix_e-_with_dist"] = low_e

# ======================================================================================
#
# Plot the fits
#
# ======================================================================================
def create_plot_name(id):
    return f"{galaxy_name}_{id:04}_size_{snapshot_size}.png"


def create_radial_profile(model_psf_bin_image, cluster_snapshot, mask, x_c, y_c):
    """
    Make a radial profile of cluster and model pixel values

    :param model_psf_bin_image: the model image on the same pixel scale as the data
    :param cluster_snapshot: the data snapshot
    :param mask: the mask indicating which pixel values to not use
    :param x_c: X coordinate of the center in snapshot coordinates
    :param y_c: Y coordinate of the center in snapshot coordinates
    :return: Three numpy arrays: The radii of all pixel values, in sorted order, the
             model values at these radii, then the data values at these radii
    """
    # When fitting I treated the center of the pixels as the integer location, so do
    # that here too
    radii, model_ys, data_ys = [], [], []
    for x in range(model_psf_bin_image.shape[1]):
        for y in range(model_psf_bin_image.shape[0]):
            if mask[y][x] > 0:
                radii.append(fit_utils.distance(x, y, x_c, y_c))
                model_ys.append(model_psf_bin_image[y, x])
                data_ys.append(cluster_snapshot[y, x])

    # sort everything in order of radii
    idxs = np.argsort(radii)
    return np.array(radii)[idxs], np.array(model_ys)[idxs], np.array(data_ys)[idxs]


def bin_profile(radii, pixel_values, bin_size):
    """
    Take an existing profile and bin it azimuthally

    :param radii: Radii values corresponding to the pixel values
    :param pixel_values: Values at the radii passed in
    :param bin_size: How big the bins should be, in pixels
    :return: Binned radii and pixel values
    """
    binned_radii, binned_ys = [], []
    for r_min in np.arange(0, int(np.ceil(max(radii))), bin_size):
        r_max = r_min + bin_size
        idx_above = np.where(r_min < radii)
        idx_below = np.where(r_max > radii)
        idx_good = np.intersect1d(idx_above, idx_below)

        if len(idx_good) > 0:
            binned_radii.append(r_min + 0.5 * bin_size)
            binned_ys.append(np.mean(pixel_values[idx_good]))

    return np.array(binned_radii), np.array(binned_ys)


def rms(sigmas, x_c, y_c, max_radius):
    """
    Calculate the RMS of the pixels within some radius

    :param sigmas: Deviations of the pixels from the fit
    :param x_c: X coordinate of the center, in coordinates of the sigmas snapshot
    :param y_c: Y coordinate of the center, in coordinates of the sigmas snapshot
    :param max_radius: Maximum radius to include in the calculation
    :return: sqrt(mean(sigmas**2)) where r < max_radius
    """
    good_sigmas = []
    for x in range(sigmas.shape[1]):
        for y in range(sigmas.shape[0]):
            if fit_utils.distance(x, y, x_c, y_c) < max_radius:
                good_sigmas.append(sigmas[y, x])

    return np.sqrt(np.mean(np.array(good_sigmas) ** 2))


def mad_of_cumulative(radii, model_cumulative, data_cumulative, max_radius):
    """
    Calculate the median relative absolute deviation of the cumulative distribution
    within some radius. This is median(abs(model - data) / data) where r < r_max

    :param radii: List of radii
    :param model_cumulative: Cumulative pixel values for the model as a function of r
    :param data_cumulative: Cumulative pixel values for the data as a function of r
    :param max_radius: Maximum radius to include in the calculation
    :return: median(abs(model - data) / data) where r < r_max
    """
    mask_good = radii < max_radius
    diffs = np.abs(model_cumulative - data_cumulative) / data_cumulative
    return np.median(diffs[mask_good])


def estimate_background(data, x_c, y_c, min_radius):
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
            if fit_utils.distance(x, y, x_c, y_c) > min_radius:
                good_bg.append(data[y, x])

    low = np.percentile(good_bg, 15.85)
    hi = np.percentile(good_bg, 84.15)
    return np.median(good_bg), 0.5 * (hi - low)


def plot_model_set(
    cluster_snapshot,
    uncertainty_snapshot,
    mask,
    params,
    id,
    r_eff,
    cut_radius,
    estimated_bg,
    bg_scatter,
):
    model_image, model_psf_image, model_psf_bin_image = fit_utils.create_model_image(
        *params, psf, snapshot_size_oversampled, oversampling_factor
    )

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
    x_c = fit_utils.oversampled_to_image(params[1], oversampling_factor)
    y_c = fit_utils.oversampled_to_image(params[2], oversampling_factor)

    radii, model_ys, data_ys = create_radial_profile(
        model_psf_bin_image, cluster_snapshot, mask, x_c, y_c
    )

    ax_pd.scatter(radii, data_ys, c=c_d, s=5, alpha=1.0, label="Data")
    ax_pd.scatter(radii, model_ys, c=c_m, s=5, alpha=1.0, label="Model")
    ax_pd.axhline(0, ls=":", c=bpl.almost_black)
    ax_pd.axvline(r_eff, ls=":", c=bpl.almost_black)

    # then bin this data to make the binned plot
    ax_pd.plot(*bin_profile(radii, data_ys, 1.0), c=c_d, lw=5, label="Binned Data")
    ax_pd.plot(*bin_profile(radii, model_ys, 1.0), c=c_m, lw=5, label="Binned Model")

    ax_pd.legend(loc="upper right")
    ax_pd.add_labels("Radius (pixels)", "Background Subtracted Pixel Value [$e^{-}$]")
    # set min and max values so it's easier to flip through bootstrapping plots
    y_min = np.min(cluster_snapshot)
    y_max = np.max(cluster_snapshot)
    # give them a bit of padding
    diff = y_max - y_min
    y_min -= 0.1 * diff
    y_max += 0.1 * diff
    ax_pd.set_limits(0, np.ceil(max(radii)), y_min, y_max)

    # then make the cumulative one. The radii are already in order so this is easy
    model_ys_cumulative = np.cumsum(model_ys)
    data_ys_cumulative = np.cumsum(data_ys)

    ax_pc.plot(radii, data_ys_cumulative, c=c_d, label="Data")
    ax_pc.plot(radii, model_ys_cumulative, c=c_m, label="Model")
    ax_pc.set_limits(0, np.ceil(max(radii)), 0, 1.2 * np.max(data_ys_cumulative))
    ax_pc.legend(loc="upper left")
    ax_pc.axvline(r_eff, ls=":", c=bpl.almost_black)
    ax_pc.add_labels(
        "Radius (pixels)", "Cumulative Background Subtracted Pixel Values [$e^{-}$]"
    )

    # calculate the relative median devation of cumulative profile and the RMS
    median_diff = mad_of_cumulative(
        radii, model_ys_cumulative, data_ys_cumulative, max_radius=cut_radius
    )
    this_rms = rms(sigma_image, snapshot_x_cen, snapshot_y_cen, cut_radius)

    # the last one just has the list of parameters
    ax_pc.easy_add_text(
        "$R_{eff}$" + f" = {r_eff:.2f} pixels\n"
        f"log(peak brightness) = {params[0]:.2f}\n"
        f"x center = {fit_utils.oversampled_to_image(params[1], oversampling_factor):.2f}\n"
        f"y center = {fit_utils.oversampled_to_image(params[2], oversampling_factor):.2f}\n"
        f"scale radius [pixels] = {params[3]:.2f}\n"
        f"q (axis ratio) = {params[4]:.2f}\n"
        f"position angle = {params[5]:.2f}\n"
        f"$\eta$ (power law slope) = {params[6]:.2f}\n"
        f"background = {params[7]:.2f}\n\n"
        f"cut radius = {cut_radius:.2f} pixels\n"
        f"estimated background = {estimated_bg:.2f}$\pm${bg_scatter:.2f}\n"
        f"RMS = {this_rms:,.2f}\n"
        f"MAD of cumulative profile = {100 * median_diff:.2f}%\n",
        "lower right",
        fontsize=15,
    )

    fig.savefig(size_dir / "cluster_fit_plots" / create_plot_name(id), dpi=100)
    plt.close(fig)
    del fig

    return median_diff, this_rms


# ======================================================================================
#
# Then the actual plotting
#
# ======================================================================================
# add the columns we want to the table
fits_catalog["profile_mad"] = -99.9
fits_catalog["estimated_local_background"] = -99.9
fits_catalog["estimated_local_background_scatter"] = -99.9
fits_catalog["estimated_local_background_diff_sigma"] = -99.9
fits_catalog["fit_rms"] = -99.9
# This is sort of copied from fit.py, as the process of getting the snapshot is
# basically the same, only here we center on the pixel where the cluster was fitted to
# be centered
for row in tqdm(fits_catalog):
    # create the snapshot. We use ceiling to get the integer pixel values as python
    # indexing does not include the final value. So when we calcualte the offset, it
    # naturally gets biased low. Moving the center up fixes that in the easiest way.
    x_cen = int(np.ceil(row["x_pix_single_fitted_best"]))
    y_cen = int(np.ceil(row["y_pix_single_fitted_best"]))

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

    snapshot_x_cen = row["x_pix_single_fitted_best"] - x_min
    snapshot_y_cen = row["y_pix_single_fitted_best"] - y_min
    mask_snapshot = fit_utils.handle_mask(mask_snapshot, snapshot_x_cen, snapshot_y_cen)

    # then get the parameters and calculate a few things of interest
    params = [
        np.log10(row["central_surface_brightness_best"]),
        fit_utils.image_to_oversampled(snapshot_x_cen, oversampling_factor),
        fit_utils.image_to_oversampled(snapshot_y_cen, oversampling_factor),
        row["scale_radius_pixels_best"],
        row["axis_ratio_best"],
        row["position_angle_best"],
        row["power_law_slope_best"],
        row["local_background_best"],
    ]

    r_eff = row["r_eff_pixels_rmax_15pix_best"]
    # Determine the radius within to calculate the fitting quality estimates
    cut_radius = np.minimum(snapshot_size / 2.0, 3 * r_eff)
    estimated_bg, bg_scatter = estimate_background(
        data_snapshot * mask_snapshot, snapshot_x_cen, snapshot_y_cen, cut_radius
    )

    profile_mad, this_rms = plot_model_set(
        data_snapshot,
        error_snapshot,
        mask_snapshot,
        params,
        row["ID"],
        r_eff,
        cut_radius,
        estimated_bg,
        bg_scatter,
    )

    row["profile_mad"] = profile_mad
    row["estimated_local_background"] = estimated_bg
    row["estimated_local_background_scatter"] = bg_scatter
    # then calculate the difference in background
    diff = row["local_background_best"] - estimated_bg
    row["estimated_local_background_diff_sigma"] = diff / bg_scatter
    row["fit_rms"] = this_rms


# ======================================================================================
#
# Then save the table
#
# ======================================================================================
# Delete the columns with distributions that we don't need anymore
fits_catalog.remove_columns(dist_cols)

fits_catalog.write(str(final_catalog_path), format="ascii.ecsv", overwrite=True)
