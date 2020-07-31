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
import numpy as np

import utils

# ======================================================================================
#
# Get the parameters the user passed in, load the catalog
#
# ======================================================================================
final_catalog_path = Path(sys.argv[1]).absolute()
fits_catalog_path = Path(sys.argv[2]).absolute()
fits_catalog = table.Table.read(fits_catalog_path, format="hdf5")

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
# Then save the table
#
# ======================================================================================
# Delete the columns with distributions that we don't need anymore
fits_catalog.remove_columns(dist_cols)

fits_catalog.write(str(final_catalog_path), format="ascii.ecsv", overwrite=True)