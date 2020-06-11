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


# ======================================================================================
#
# Convert pixels to pc
#
# ======================================================================================
pixels_to_pc = utils.get_f555w_pixel_scale_pc(final_catalog_path.parent.parent)
a_pc = "scale_radius_pc"
a_pix = "scale_radius_pixels"
fits_catalog[f"{a_pc}_best"] = fits_catalog[f"{a_pix}_best"] * pixels_to_pc
fits_catalog[a_pc] = fits_catalog[a_pix] * pixels_to_pc


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
r_eff = "effective_radius_pc"
eta = fits_catalog["power_law_slope"]
a = fits_catalog["scale_radius_pc"]
q = fits_catalog["axis_ratio"]
fits_catalog[f"{r_eff}_no_rmax"] = eff_profile_r_eff_no_rmax(eta, a, q)
fits_catalog[f"{r_eff}_rmax_100_pc"] = eff_profile_r_eff_with_rmax(eta, a, 100, q)
# add one with the size of the radius

# ======================================================================================
#
# Then save the table
#
# ======================================================================================
fits_catalog.write(str(final_catalog_path), format="hdf5", overwrite=True)
