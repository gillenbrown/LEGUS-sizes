"""
artificial_comparison.py
Compare the results of the artificial cluster test to the true results
"""

import sys
from pathlib import Path
import numpy as np
from astropy import table
import betterplotlib as bpl

bpl.set_style()

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1]).resolve()
catalog_name = Path(sys.argv[2]).resolve()
catalog = table.Table.read(catalog_name, format="ascii.ecsv")
catalog["reff_pixels_true"] = 0.0

# ======================================================================================
#
# copy some functions to get the true effective radius
#
# ======================================================================================
# I can't import these easily, unfortunately
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
    Correction for ellipticity. This gives R_eff(q) / R_eff(q=1)

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


# then add this effective radius
for row in catalog:
    row["reff_pixels_true"] = eff_profile_r_eff_with_rmax(
        row["power_law_slope_true"],
        row["scale_radius_pixels_true"],
        row["axis_ratio_true"],
        15,  # size of the box
    )

# ======================================================================================
#
# Then calculate the error for the parameters of interest
#
# ======================================================================================
reff = catalog["r_eff_pixels_rmax_15pix_best"]
reff_true = catalog["reff_pixels_true"]
max_err = np.maximum(
    catalog["r_eff_pixels_rmax_15pix_e+"], catalog["r_eff_pixels_rmax_15pix_e-"]
)
catalog["reff_sigma_error"] = np.abs(reff - reff_true) / max_err
catalog["reff_relative_error"] = np.abs(reff - reff_true) / reff_true

# ======================================================================================
#
# Then plot this up
#
# ======================================================================================
fig, ax = bpl.subplots()
ax.scatter(reff_true, reff, alpha=1)
ax.add_labels("True $R_{eff}$ [pixels]", "Measured $R_{eff}$ [pixels]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.equal_scale()
ax.set_limits(0.05, 30, 0.05, 30)
ax.plot([1e-5, 100], [1e-5, 100], ls=":", c=bpl.almost_black, zorder=0)
fig.savefig(plot_name)
