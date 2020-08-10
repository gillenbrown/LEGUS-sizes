"""
experiments.py - script that holds temporary experimental things

This is set up this way so that I can easily pass in all the catalogs
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits
from scipy import special
from matplotlib import colors, cm
import betterplotlib as bpl
import cmocean

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
galaxy = "ngc628-c"
cluster_id = 778

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

# Then get the snapshot of this cluster
x_cen = int(np.ceil(row["x_pix_single_fitted_best"]))
y_cen = int(np.ceil(row["y_pix_single_fitted_best"]))

# Get the snapshot, based on the size desired
x_min = x_cen - 15
x_max = x_cen + 15
y_min = y_cen - 15
y_max = y_cen + 15

data_snapshot = image_data[y_min:y_max, x_min:x_max]
error_snapshot = error_data[y_min:y_max, x_min:x_max]
mask_snapshot = mask_data[y_min:y_max, x_min:x_max]

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
    prior = priors(*params)
    if prior > 0:
        log_prior = np.log(prior)
    else:
        # infinity messes up the scipy fitting routine, a large finite value is better
        log_prior = -1e100
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
# Then calculate this for lots of values
# ======================================================================================
# use the best fit parameters when we're not varying eta
log_mu_0 = np.log10(row["central_surface_brightness_best"])
x_c = row["x_pix_snapshot_oversampled_best"]
y_c = row["y_pix_snapshot_oversampled_best"]
q = row["axis_ratio_best"]
theta = row["position_angle_best"]
background = row["local_background_best"]

# change these
eta_min, eta_max, eta_d = (0, 3, 0.025)
alog_min, alog_max, alog_d = (-5, 1, 0.05)
# don't mess with this
eta_values = np.arange(eta_min, eta_max + 0.5 * eta_d, eta_d)
a_values = 10 ** (np.arange(alog_min, alog_max + 0.5 * alog_d, alog_d))

n_eta = len(eta_values)
n_a = len(a_values)
# then make the output array
output = np.zeros((n_a, n_eta))

# then fill the output array
for idx_eta in range(n_eta):
    for idx_a in range(n_a):
        eta = eta_values[idx_eta]
        a = a_values[idx_a]

        params = (log_mu_0, x_c, y_c, a, q, theta, eta, background)

        neg_log_likelihood = negative_log_likelihood(
            params, data_snapshot, error_snapshot, mask_snapshot
        )
        output[idx_a, idx_eta] = neg_log_likelihood

# Mark the point of maximum likelihood as identified by the fit. The coordinates are
# indices, so we basically do what we did before. We'll put a marker at the center of
# the best fit cell, I won't bother doing trying to figure out where within the cell
best_eta = row["power_law_slope_best"]
best_a = row["scale_radius_pixels_best"]
for idx_eta in range(n_eta - 1):
    if eta_values[idx_eta] < best_eta <= eta_values[idx_eta + 1]:
        eta_idx_best = idx_eta
        break
for idx_a in range(n_a - 1):
    if a_values[idx_a] < best_a <= a_values[idx_a + 1]:
        a_idx_best = idx_a
        break

# ======================================================================================
# plot
# ======================================================================================
def format_a(a):
    exponent = np.log10(a)
    if a > 101 or a < 0.00999:
        return "$10^{" + f"{exponent:.2g}" + "}$"
    if np.isclose(a, 100, rtol=0, atol=0.01):
        return "100"
    else:
        return f"{a:g}"


fig, ax = bpl.subplots()
vmin = np.min(output)
width = 5
norm = colors.Normalize(vmin=vmin, vmax=width + vmin, clip=True)
cmap = cmocean.cm.haline_r
i = ax.imshow(output, origin="lower", norm=norm, cmap=cmap)
cbar = fig.colorbar(i, ax=ax)
cbar.set_label("-log$_{10}$(Likelihood) = $\chi^2 -$ log$_{10}(P(\\theta))$")
# mark the best fit point
ax.scatter([eta_idx_best], [a_idx_best], marker="x", c=bpl.almost_black)

# # Also draw contours
# # level_factors = [2, 10, 100, 1000]
# # levels = [l_f * vmin for l_f in level_factors]
# levels = [100, 1000, 1e4, 1e5]
# ax.contour(
#     eta_values,
#     a_values,
#     output,
#     levels=levels,
#     # colors=bpl.almost_black,
#     zorder=100,
#     origin="lower",
# )

tick_gap_eta = 20
tick_gap_a = 20
ax.xaxis.set_ticks(range(0, n_eta, tick_gap_eta))
ax.yaxis.set_ticks(range(0, n_a, tick_gap_a))
ax.xaxis.set_ticklabels([f"{item:.2g}" for item in eta_values[::tick_gap_eta]])
ax.yaxis.set_ticklabels([format_a(item) for item in a_values[::tick_gap_a]])
ax.add_labels("$\eta$ (Power Law Slope)", "a (Scale Radius) [pixels]")
ax.easy_add_text(f"{galaxy.upper()} - {cluster_id}", "upper left", color="white")
fig.savefig(Path(__file__).parent / "likelihood_contours.png", bbox_inches="tight")

# ======================================================================================
#
# Do not remove this line!!
#
# ======================================================================================
sentinel_name.touch()
