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
# Convert pixels to pc
#
# ======================================================================================
pixels_to_pc = utils.get_f555w_pixel_scale_pc(final_catalog_path.parent.parent)
a_pc = "scale_radius_pc"
a_pix = "scale_radius_pixels"
fits_catalog[f"{a_pc}_best"] = fits_catalog[f"{a_pix}_best"] * pixels_to_pc
fits_catalog[a_pc] = [row[a_pix] * pixels_to_pc for row in fits_catalog]

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


def eff_profile_r_eff_with_rmax(eta, a, rmax, q):
    """
    Calculate the effective radius of an EFF profile, assuming a maximum radius.

    :param eta: Power law slope of the EFF profile
    :param a: Scale radius of the EFF profile, in any units.
    :param rmax: Maximum radius for the profile, in the same units as a.
    :return: Effective radius, in the same units as a and rmax
    """
    # This is such an ugly formula, put it in a few steps
    term_1 = 1 + (1 + (rmax / a) ** 2) ** (1 - eta)
    term_2 = (0.5 * (term_1)) ** (1 / (1 - eta)) - 1
    return ellipticy_correction(q) * a * np.sqrt(term_2)


# ======================================================================================
#
# Then add these quantities to the table
#
# ======================================================================================
# add the columns we want to the table
fits_catalog["r_eff_pc_rmax_100pc_e+"] = -99.9
fits_catalog["r_eff_pc_rmax_100pc_e-"] = -99.9

# first calculate the best fit value
fits_catalog["r_eff_pc_rmax_100pc_best"] = eff_profile_r_eff_with_rmax(
    fits_catalog["power_law_slope_best"],
    fits_catalog["scale_radius_pc_best"],
    fits_catalog["axis_ratio_best"],
    100,
)

# calculate the distribution of all effective radii in each case
for row in fits_catalog:
    eta = row["power_law_slope"]
    a = row["scale_radius_pc"]
    q = row["axis_ratio"]
    all_r_eff_no_max = eff_profile_r_eff_with_rmax(eta, a, q, 100)

    low, hi = np.percentile(all_r_eff_no_max, [15.85, 84.15])
    row["r_eff_pc_rmax_100pc_e+"] = hi - row["r_eff_pc_rmax_100pc_best"]
    row["r_eff_pc_rmax_100pc_e-"] = row["r_eff_pc_rmax_100pc_best"] - low

# ======================================================================================
#
# Then save the table
#
# ======================================================================================
# Delete the columns with distributions that we don't need anymore
fits_catalog.remove_columns(dist_cols + ["scale_radius_pc"])

fits_catalog.write(str(final_catalog_path), format="ascii.ecsv", overwrite=True)
