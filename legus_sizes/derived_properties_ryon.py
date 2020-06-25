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
def ellipticy_correction(q):
    return (1 + q) / 2.0


def eff_profile_r_eff_no_rmax(eta, a, q):
    """
    Calculate the effective radius of an EFF profile assuming to maximum radius

    :param eta: Power law slope of the EFF profile
    :param a: Scale radius of the EFF profile.
    :return: Effective radius of the EFF profile, in whatever units a is in.
    """
    return ellipticy_correction(q) * a * np.sqrt(2 ** (1 / (eta - 1)) - 1)


# ======================================================================================
#
# Then add these quantities to the table
#
# ======================================================================================
# add the columns we want to the table
fits_catalog["r_eff_pixels_no_rmax_e+"] = -99.9
fits_catalog["r_eff_pixels_no_rmax_e-"] = -99.9
fits_catalog["r_eff_pc_no_rmax_best"] = -99.9
fits_catalog["r_eff_pc_no_rmax_e+"] = -99.9
fits_catalog["r_eff_pc_no_rmax_e-"] = -99.9
fits_catalog["r_eff_pc_no_rmax_e+_with_dist"] = -99.9
fits_catalog["r_eff_pc_no_rmax_e-_with_dist"] = -99.9

# first calculate the best fit value
fits_catalog["r_eff_pixels_no_rmax_best"] = eff_profile_r_eff_no_rmax(
    fits_catalog["power_law_slope_best"],
    fits_catalog["scale_radius_pixels_best"],
    fits_catalog["axis_ratio_best"],
)

# calculate the distribution of all effective radii in each case
for row in fits_catalog:
    # Calculate the effective radius in pixels for all bootstrapping iterations
    all_r_eff_pixels = eff_profile_r_eff_no_rmax(
        row["power_law_slope"], row["scale_radius_pixels"], row["axis_ratio"]
    )
    # then get the 1 sigma error range of that
    low, hi = np.percentile(all_r_eff_pixels, [15.85, 84.15])
    # subtract the middle to get the error range
    row["r_eff_pixels_no_rmax_e+"] = max(hi - row["r_eff_pixels_no_rmax_best"], 0)
    row["r_eff_pixels_no_rmax_e-"] = max(row["r_eff_pixels_no_rmax_best"] - low, 0)

    # Then we can convert to pc. First do it without including distance errors
    best, low_e, high_e = utils.pixels_to_pc_with_errors(
        final_catalog_path.parent.parent,
        row["r_eff_pixels_no_rmax_best"],
        row["r_eff_pixels_no_rmax_e-"],
        row["r_eff_pixels_no_rmax_e+"],
        include_distance_err=False,
    )

    row["r_eff_pc_no_rmax_best"] = best
    row["r_eff_pc_no_rmax_e+"] = high_e
    row["r_eff_pc_no_rmax_e-"] = low_e

    # Then recalculate the errors including the distance errors
    _, low_e, high_e = utils.pixels_to_pc_with_errors(
        final_catalog_path.parent.parent,
        row["r_eff_pixels_no_rmax_best"],
        row["r_eff_pixels_no_rmax_e-"],
        row["r_eff_pixels_no_rmax_e+"],
        include_distance_err=True,
    )

    row["r_eff_pc_no_rmax_e+_with_dist"] = high_e
    row["r_eff_pc_no_rmax_e-_with_dist"] = low_e

# ======================================================================================
#
# Then save the table
#
# ======================================================================================
# Delete the columns with distributions that we don't need anymore
fits_catalog.remove_columns(dist_cols)

fits_catalog.write(str(final_catalog_path), format="ascii.ecsv", overwrite=True)
