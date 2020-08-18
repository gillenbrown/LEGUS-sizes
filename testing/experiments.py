"""
experiments.py - script that holds temporary experimental things

This is set up this way so that I can easily pass in all the catalogs
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits
from scipy import special, optimize
from matplotlib import colors, cm, ticker
import betterplotlib as bpl
import cmocean
from tqdm import tqdm

# need to add the correct path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
import utils, fit_utils

bpl.set_style()

# ======================================================================================
#
# load the parameters the user passed in
#
# ======================================================================================
sentinel_name = Path(sys.argv[1])
catalogs = []
for item in sys.argv[2:]:
    cat = table.Table.read(item, format="ascii.ecsv")
    cat["galaxy"] = Path(item).parent.parent.name
    catalogs.append(cat)
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# ======================================================================================
#
# Experiments start here
#
# ======================================================================================
# I examined some fits with open_some_fits.py and found an example with a good
# background value, low eta and a, and that's relatively isolated. These attributes
# ensure that the fit is otherwise good except for unrealistic values for a and eta.
# This means we can modify those two parameters and see the chi-squared surface easily
# galaxy = "ngc4656"
# cluster_id = 22

# galaxy = "ngc628-c"
# cluster_id = 1630

# galaxy = "ngc628-c"
# cluster_id = 778

# galaxy = "ic559"
# cluster_id = 1

galaxy = "ic559"
cluster_id = 9

# galaxy = "ic559"
# cluster_id = 12

# galaxy = "ngc1313-e"
# cluster_id = 1813

# galaxy = "ngc1313-e"
# cluster_id = 118

# galaxy = "ngc1433"
# cluster_id = 83

for row in big_catalog:
    if row["galaxy"] == galaxy and row["ID"] == cluster_id:
        break
# for now get the first galaxy that passes this criterion
print(
    row["galaxy"],
    row["ID"],
    row["power_law_slope_best"],
    row["scale_radius_pixels_best"],
    "\n",
)

# some other stuff needed
psf_size = 15
snapshot_size = 30
oversampling_factor = 2
snapshot_size_oversampled = snapshot_size * oversampling_factor

# Load the data needed to calculate chi-squared
gal_dir = Path(__file__).parent.parent / "data" / galaxy
image_data, _, _ = utils.get_drc_image(gal_dir)
error_data = fits.open(gal_dir / "size" / "sigma_electrons.fits")["PRIMARY"].data
mask_data = fits.open(gal_dir / "size" / "mask_image.fits")["PRIMARY"].data

# get the PSF for this galaxy
psf_name = f"psf_my_stars_{psf_size}_pixels_{oversampling_factor}x_oversampled.fits"
psf = fits.open(gal_dir / "size" / psf_name)["PRIMARY"].data

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
error_snapshot = error_data[y_min:y_max, x_min:x_max].copy()
mask_snapshot = mask_data[y_min:y_max, x_min:x_max].copy()

# process the mask
mask_snapshot = fit_utils.handle_mask(mask_snapshot, cluster_id)

# have the start value for the fit (which will be the same as actually used in the fit)
bg_start = np.min(data_snapshot)
mu_start = np.log10(np.max(data_snapshot) * 3)

# use the best fit parameters when we're not varying eta
log_mu_0 = np.log10(row["central_surface_brightness_best"])
x_c = row["x_pix_snapshot_oversampled_best"]
y_c = row["y_pix_snapshot_oversampled_best"]
q = row["axis_ratio_best"]
theta = row["position_angle_best"]
background = row["local_background_best"]
estimated_bg = row["estimated_local_background"]
estimated_bg_scatter = row["estimated_local_background_scatter"]

# ======================================================================================
# Functions to calculate chi-squared
# ======================================================================================
# These functions are copied from fit.py
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
    return np.sum(sigma_snapshot ** 2)


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


def flat_with_normal_edges(x, lower_edge, upper_edge, side_width, boundary_value):
    """
    Probability density function that is flat with value at 1 in between two
    values, but with a lognormal PDF outside of that, with fixed width on both
    sides. This returns the log of that density.

    This is useful as it is continuous and smooth at the boundaries of the
    flat region.

    :param x: Value to determine the value of the PDF at.
    :param lower_edge: Lower value (see `mode`)
    :param upper_edge: Upper value (see `mode`)
    :param side_log_width: Gaussian width (in dex) of the normal distribution
                           used for the regions on either side.
    :param boundary_value: The value that the pdf should take at the edges. If this is
                           1.0, the flat region will extend to the edges. However, if
                           another is value is preferred at those edges, the flat
                           region will shrink to make room for that.
    :return:
    """
    lower_mean = mean_of_normal(lower_edge, boundary_value, side_width, False)
    upper_mean = mean_of_normal(upper_edge, boundary_value, side_width, True)

    if lower_edge > lower_edge:
        raise ValueError("There is not flat region with these parameters.")

    # check that we did the math right for the shape of the distribution
    assert np.isclose(
        np.exp(log_of_normal(lower_edge, lower_mean, side_width)), boundary_value
    )
    assert np.isclose(
        np.exp(log_of_normal(upper_edge, upper_mean, side_width)), boundary_value
    )

    if lower_mean <= x <= upper_mean:
        return 0
    elif x < lower_mean:
        return log_of_normal(x, lower_mean, side_width)
    else:
        return log_of_normal(x, upper_mean, side_width)


def log_priors(log_mu_0, x_c, y_c, a, q, theta, eta, background):
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
    log_prior = 0
    # prior are multiplicative, or additive in log space
    log_prior += flat_with_normal_edges(
        np.log10(a), np.log10(0.1), np.log10(15), 0.1, 0.5
    )
    log_prior += flat_with_normal_edges(q, 0.3, 1.0, 0.1, 1.0)
    return log_prior


def postprocess_params(log_mu_0, x_c, y_c, a, q, theta, eta, background):
    """
    Postprocess the parameters, namely the axis ratio and position angle.

    This is needed since we let the fit have axis ratios larger than 1, and position
    angles of any value. Axis ratios larger than 1 indicate that we need to flip the
    major and minor axes. This requires rotating the position angle 90 degrees, and
    shifting the value assigned to the major axis to correct for the improper axis
    ratio.
    """
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


def negative_log_likelihood(params, cluster_snapshot, error_snapshot, mask, use_prior):
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
    log_data_likelihood = -chi_sq / 2.0
    # Need to postprocess the parameters before calculating the prior, as the prior
    # is on the physically reasonable values, we need to make sure that's correct.
    if use_prior:
        log_prior = log_priors(*postprocess_params(*params))
    else:
        log_prior = 0
    log_likelihood = log_data_likelihood + log_prior
    assert not np.isnan(log_prior)
    assert not np.isnan(log_data_likelihood)
    assert not np.isinf(log_prior)
    assert not np.isinf(log_data_likelihood)
    assert not np.isneginf(log_prior)
    assert not np.isneginf(log_data_likelihood)
    # return the negative of this so we can minimize this value
    return -log_likelihood


# ======================================================================================
# Some plot functions
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


def eff_profile_r_eff_with_rmax(a, eta, q, rmax):
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


def eta_given_reff_rmax_a(r_eff, r_max, a, q):
    def to_minimize(eta, r_eff, r_max, a, q):
        return abs(r_eff - eff_profile_r_eff_with_rmax(a, eta, q, r_max))

    return optimize.minimize(
        to_minimize, x0=(1e-15,), args=(r_eff, r_max, a, q), bounds=[(0, None),]
    ).x


# generate a list of lines of constant effective radius
def generate_r_eff_lines(r_eff_values, a_values, q):
    return_dict = {r: [[], []] for r in r_eff_values}
    for r in tqdm(r_eff_values):
        for a in a_values:
            eta = eta_given_reff_rmax_a(r, 15, a, q)

            r_eff_calculated = eff_profile_r_eff_with_rmax(a, eta, q, 15)
            if np.isclose(r_eff_calculated, r):
                return_dict[r][0].append(eta)
                return_dict[r][1].append(np.log10(a))

    return return_dict


# generate the lines of constant effective radius to plot
r_eff_a_values = np.logspace(-6, 2, 100)
r_eff_lines = generate_r_eff_lines([0.1, 1, 5, 10], r_eff_a_values, q)


def format_exponent(log_a, pos):
    assert np.isclose(float(log_a), int(log_a))
    log_a = int(log_a)

    if log_a > -2:
        return str(10 ** log_a)
    else:
        return "$10^{" + f"{log_a}" + "}$"


likelihood_cmap = cmocean.cm.haline

# # ======================================================================================
# # Then calculate this for lots of values
# # ======================================================================================
# # change these
# eta_min, eta_max, d_eta = (0, 3, 0.025)
# log_a_min, log_a_max, d_log_a = (-5, 1, 0.05)
# # don't mess with this
# eta_values = np.arange(eta_min, eta_max + 0.5 * d_eta, d_eta)
# log_a_values = np.arange(log_a_min, log_a_max + 0.5 * d_log_a, d_log_a)
# a_values = 10 ** log_a_values
#
# n_eta = len(eta_values)
# n_a = len(a_values)
# # then make the output arrays
# log_likelihood_fixed_params = np.zeros((n_a, n_eta))
#
# # then fill the output array
# for idx_eta in tqdm(range(n_eta)):
#     for idx_a in range(n_a):
#         eta = eta_values[idx_eta]
#         a = a_values[idx_a]
#
#         params = (log_mu_0, x_c, y_c, a, q, theta, eta, background)
#
#         log_likelihood = -1 * negative_log_likelihood(
#             params, data_snapshot, error_snapshot, mask_snapshot
#         )
#         log_likelihood_fixed_params[idx_a, idx_eta] = log_likelihood
#
# # ======================================================================================
# # plot
# # ======================================================================================
# fig, ax = bpl.subplots()
#
# vmax = np.max(log_likelihood_fixed_params)
# width = 2
# norm = colors.Normalize(vmin=vmax - width, vmax=vmax, clip=True)
# limits = (
#     eta_min - 0.5 * d_eta,
#     eta_max + 0.5 * d_eta,
#     log_a_min - 0.5 * d_log_a,
#     log_a_max + 0.5 * d_log_a,
# )
# i = ax.imshow(
#     log_likelihood_fixed_params,
#     origin="lower",
#     norm=norm,
#     cmap=likelihood_cmap,
#     extent=limits,
#     # This scalar aspect ratio is calculated to ensure the pixels are square
#     aspect=((eta_max - eta_min) / n_eta) / ((log_a_max - log_a_min) / n_a),
# )
# cbar = fig.colorbar(i, ax=ax)
# cbar.set_label("Log Likelihood = $-\chi^2 +$ log$_{10}(P(\\theta))$")
# # mark the best fit point
# ax.scatter(
#     [row["power_law_slope_best"]],
#     [np.log10(row["scale_radius_pixels_best"])],
#     marker="x",
#     c=bpl.almost_black,
# )
#
# # Also draw contours
# levels = [vmax - 1e20, vmax - 1]
# contour = ax.contour(
#     eta_values,
#     log_a_values,
#     log_likelihood_fixed_params,
#     levels=levels,
#     colors="red",
#     linestyles="solid",
#     linewidths=2,
#     origin="lower",
# )
#
# # and plot lines of effective radius
# for r in r_eff_lines:
#     xs = r_eff_lines[r][0]
#     ys = r_eff_lines[r][1]
#     ax.plot(xs, ys, c="w")
#     # pick the first value that goes above the lower limit
#     for idx, y in enumerate(ys):
#         if y > log_a_min:
#             break
#     ax.add_text(
#         x=xs[idx],
#         y=ys[idx],
#         text="$R_{eff}=$" + f"{r:.1f} pixels",
#         ha="right",
#         va="bottom",
#         fontsize=12,
#         rotation=90,
#         color="w",
#     )
#
# # plot Oleg's guess.
# guess_log_as = -3 / eta_values
# ax.plot(eta_values, guess_log_as, c="violet", zorder=10)
#
# ax.set_limits(*limits)
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_exponent))
# ax.add_labels("$\eta$ (Power Law Slope)", "a (Scale Radius) [pixels]")
# ax.easy_add_text(f"{galaxy.upper()} - {cluster_id}", "upper left", color="white")
# fig.savefig(
#     Path(__file__).parent / f"likelihood_contours_{galaxy}_{cluster_id}.png",
#     bbox_inches="tight",
# )

# # ======================================================================================
# # Fit the background and peak value at each point
# # ======================================================================================
# # change these
# eta_min, eta_max, d_eta = (0, 3, 0.5)
# log_a_min, log_a_max, d_log_a = (-5, 1, 1)
# # don't mess with this
# eta_values = np.arange(eta_min, eta_max + 0.5 * d_eta, d_eta)
# log_a_values = np.arange(log_a_min, log_a_max + 0.5 * d_log_a, d_log_a)
# a_values = 10 ** log_a_values
#
# n_eta = len(eta_values)
# n_a = len(a_values)
#
# # then make the output arrays
# log_likelihood_fitted_params = np.zeros((n_a, n_eta))
# bg_err = np.zeros((n_a, n_eta))
#
# for idx_eta in tqdm(range(n_eta)):
#     for idx_a in range(n_a):
#         eta = eta_values[idx_eta]
#         a = a_values[idx_a]
#         # try to mess with the other parameters tosee if we can do better
#         params = (x_c, y_c, a, q, theta, eta)
#
#         def chi_sq_wrapper(
#             to_fit, params, data_snapshot, error_snapshot, mask_snapshot
#         ):
#             return negative_log_likelihood(
#                 (to_fit[1],) + params + (to_fit[0],),
#                 data_snapshot,
#                 error_snapshot,
#                 mask_snapshot,
#             )
#
#         fit_params = optimize.minimize(
#             chi_sq_wrapper,
#             x0=(bg_start, mu_start),
#             args=(params, data_snapshot, error_snapshot, mask_snapshot),
#             bounds=((None, None), (None, 100)),
#         ).x
#         bg_fit = fit_params[0]
#         log_mu = fit_params[1]
#
#         log_likelihood_fitted_params[idx_a, idx_eta] = -1 * negative_log_likelihood(
#             (log_mu,) + params + (bg_fit,),
#             data_snapshot,
#             error_snapshot,
#             mask_snapshot,
#         )
#         bg_err[idx_a, idx_eta] = (bg_fit - estimated_bg) / estimated_bg_scatter
#
# # ======================================================================================
# # plot the fitted background
# # ======================================================================================
#
# fig, axs = bpl.subplots(ncols=2, figsize=[15, 6])
#
# likelihood_width = 2
# likelihood_norm = colors.Normalize(
#     vmin=np.max(log_likelihood_fitted_params) - likelihood_width,
#     vmax=np.max(log_likelihood_fitted_params),
#     clip=True,
# )
# limits = (
#     eta_min - 0.5 * d_eta,
#     eta_max + 0.5 * d_eta,
#     log_a_min - 0.5 * d_log_a,
#     log_a_max + 0.5 * d_log_a,
# )
# i = axs[0].imshow(
#     log_likelihood_fitted_params,
#     origin="lower",
#     extent=limits,
#     norm=likelihood_norm,
#     cmap=likelihood_cmap,
#     # This scalar aspect ratio is calculated to ensure the pixels are square
#     aspect=((eta_max - eta_min) / n_eta) / ((log_a_max - log_a_min) / n_a),
# )
# cbar = fig.colorbar(i, ax=axs[0])
# cbar.set_label("Log Likelihood = $-\chi^2 +$ log$_{10}(P(\\theta))$")
# # Also draw contours
# levels = [
#     np.max(log_likelihood_fitted_params) - 1e20,
#     np.max(log_likelihood_fitted_params) - 1,
# ]
# contour = axs[0].contour(
#     eta_values,
#     log_a_values,
#     log_likelihood_fitted_params,
#     levels=levels,
#     colors="red",
#     linestyles="solid",
#     linewidths=2,
#     origin="lower",
# )
#
# bg_norm = colors.Normalize(-3, 3, clip=True)
# bg_cmap = cmocean.cm.curl
# i = axs[1].imshow(
#     bg_err,
#     origin="lower",
#     extent=limits,
#     norm=bg_norm,
#     cmap=bg_cmap,
#     # This scalar aspect ratio is calculated to ensure the pixels are square
#     aspect=((eta_max - eta_min) / n_eta) / ((log_a_max - log_a_min) / n_a),
# )
# cbar = fig.colorbar(i, ax=axs[1])
# cbar.set_label("Background Error")
# # mark the best fit point
# for c, ax in zip([bpl.almost_black, "w"], axs):
#     ax.scatter(
#         [row["power_law_slope_best"]],
#         [np.log10(row["scale_radius_pixels_best"])],
#         marker="x",
#         c=c,
#     )
#
# # and plot lines of effective radius
# for ax in axs:
#     for r in r_eff_lines:
#         xs = r_eff_lines[r][0]
#         ys = r_eff_lines[r][1]
#         ax.plot(xs, ys, c="w")
#         # pick the first value that goes above the lower limit
#         for idx, y in enumerate(ys):
#             if y > log_a_min:
#                 break
#         ax.add_text(
#             x=xs[idx],
#             y=ys[idx],
#             text="$R_{eff}=$" + f"{r:.1f} pixels",
#             ha="right",
#             va="bottom",
#             fontsize=12,
#             rotation=90,
#             color="w",
#         )
#
# for ax in axs:
#     ax.set_limits(*limits)
#     ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_exponent))
#     ax.add_labels("$\eta$ (Power Law Slope)", "a (Scale Radius) [pixels]")
#     ax.easy_add_text(f"{galaxy.upper()} - {cluster_id}", "upper left", color="white")
# fig.savefig(
#     Path(__file__).parent / f"likelihood_contours_fit_bg_{galaxy}_{cluster_id}.png",
#     bbox_inches="tight",
# )

# ======================================================================================
# Fit the background and peak value at each point, comparing with and without priors
# ======================================================================================
# change these
eta_min, eta_max, d_eta = (0, 3, 0.5)
log_a_min, log_a_max, d_log_a = (-5, 1, 1.0)
# don't mess with this
eta_values = np.array(
    [row["power_law_slope_best"] + k * d_eta for k in range(-100, 100)]
)
eta_values = eta_values[eta_values >= eta_min]
eta_values = eta_values[eta_values <= eta_max]
log_a_values = np.array(
    [np.log10(row["scale_radius_pixels_best"]) + k * d_log_a for k in range(-100, 100)]
)
log_a_values = log_a_values[log_a_values >= log_a_min]
log_a_values = log_a_values[log_a_values <= log_a_max]
# eta_values = np.arange(eta_min, eta_max + 0.5 * d_eta, d_eta)
# log_a_values = np.arange(log_a_min, log_a_max + 0.5 * d_log_a, d_log_a)
a_values = 10 ** log_a_values

n_eta = len(eta_values)
n_a = len(a_values)

# then make the output arrays
log_likelihood_with_prior = np.zeros((n_a, n_eta))
log_likelihood_no_prior = np.zeros((n_a, n_eta))

for idx_eta in tqdm(range(n_eta)):
    for idx_a in range(n_a):
        eta = eta_values[idx_eta]
        a = a_values[idx_a]
        # try to mess with the other parameters tosee if we can do better
        params = (x_c, y_c, a, q, theta, eta)

        def chi_sq_wrapper(
            to_fit, params, data_snapshot, error_snapshot, mask_snapshot, use_priors
        ):
            return negative_log_likelihood(
                (to_fit[1],) + params + (to_fit[0],),
                data_snapshot,
                error_snapshot,
                mask_snapshot,
                use_priors,
            )

        for use_prior in [True, False]:
            # Find the best fit parameters
            fit_params = optimize.minimize(
                chi_sq_wrapper,
                x0=(bg_start, mu_start),
                args=(params, data_snapshot, error_snapshot, mask_snapshot, use_prior),
                bounds=((None, None), (None, 100)),
            ).x
            bg_fit = fit_params[0]
            log_mu = fit_params[1]

            # Then plug these into the likelihood function to see what we get
            this_log_likelihood = -1 * negative_log_likelihood(
                (log_mu,) + params + (bg_fit,),
                data_snapshot,
                error_snapshot,
                mask_snapshot,
                use_prior,
            )

            if use_prior:
                log_likelihood_with_prior[idx_a, idx_eta] = this_log_likelihood
            else:
                log_likelihood_no_prior[idx_a, idx_eta] = this_log_likelihood

# ======================================================================================
# plot this
# ======================================================================================

fig, axs = bpl.subplots(ncols=2, figsize=[15, 6])

likelihood_width = 200
vmax = max(np.max(log_likelihood_no_prior), np.max(log_likelihood_with_prior))
likelihood_norm = colors.Normalize(vmin=vmax - likelihood_width, vmax=vmax, clip=True)

for ax, log_likelihood_fitted_params in zip(
    axs, [log_likelihood_no_prior, log_likelihood_with_prior]
):
    limits = (
        min(eta_values) - 0.5 * d_eta,
        max(eta_values) + 0.5 * d_eta,
        min(log_a_values) - 0.5 * d_log_a,
        max(log_a_values) + 0.5 * d_log_a,
    )
    i = ax.imshow(
        log_likelihood_fitted_params,
        origin="lower",
        extent=limits,
        norm=likelihood_norm,
        cmap=likelihood_cmap,
        # This scalar aspect ratio is calculated to ensure the pixels are square
        aspect=((eta_max - eta_min) / n_eta) / ((log_a_max - log_a_min) / n_a),
    )
    cbar = fig.colorbar(i, ax=ax)
    cbar.set_label("Log Likelihood = $-\chi^2/2 +$ ln$(P(\\theta))$")

    ax.set_limits(*limits)

# mark the best fit point
for ax in axs:
    ax.scatter(
        [row["power_law_slope_best"]],
        [np.log10(row["scale_radius_pixels_best"])],
        marker="x",
        c=bpl.almost_black,
    )

axs[0].set_title("No Priors")
axs[1].set_title("With Priors")

for ax in axs:
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_exponent))
    ax.add_labels("$\eta$ (Power Law Slope)", "a (Scale Radius) [pixels]")
    ax.easy_add_text(f"{galaxy.upper()} - {cluster_id}", "upper left", color="white")
fig.savefig(
    Path(__file__).parent / f"likelihood_contours_priors_{galaxy}_{cluster_id}.png",
    bbox_inches="tight",
)
# # ======================================================================================
#
# Do not remove this line!!
#
# ======================================================================================
sentinel_name.touch()
