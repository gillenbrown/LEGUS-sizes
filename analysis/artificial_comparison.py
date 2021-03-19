"""
artificial_comparison.py
Compare the results of the artificial cluster test to the true results
"""

import sys
from pathlib import Path
import numpy as np
from astropy import table
import cmocean
from matplotlib import ticker, colors, cm
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
# Function to use to set the ticks
@ticker.FuncFormatter
def nice_log_formatter(x, pos):
    exp = np.log10(x)
    # this only works for labels that are factors of 10. Other values will produce
    # misleading results, so check this assumption.
    assert np.isclose(exp, int(exp))

    # for values between 0.01 and 100, just use that value.
    # Otherwise use the log.
    if abs(exp) < 2:
        return f"{x:g}"
    else:
        return "$10^{" + f"{exp:.0f}" + "}$"


cmap = cmocean.cm.thermal_r
cmap = cmocean.tools.crop_by_percent(cmap, 15, "both")
boundaries = np.arange(0.75, 2.7501, 0.5)
norm = colors.BoundaryNorm(
    boundaries,
    ncolors=256,
)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
plot_colors = [mappable.to_rgba(eta) for eta in catalog["power_law_slope_true"]]


fig, ax = bpl.subplots()
s = ax.scatter(
    reff_true,
    reff,
    alpha=1,
    c=plot_colors,
)
ax.plot([1e-5, 100], [1e-5, 100], ls=":", c=bpl.almost_black, zorder=0)
ax.add_labels("True $R_{eff}$ [pixels]", "Measured $R_{eff}$ [pixels]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.equal_scale()
ax.set_limits(0.001, 10, 0.001, 10)
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.xaxis.set_major_formatter(nice_log_formatter)
ax.yaxis.set_major_formatter(nice_log_formatter)
cbar = fig.colorbar(mappable, ax=ax)
cbar.set_label("Power Law Slope $\eta$")
cbar.set_ticks(sorted(np.unique(catalog["power_law_slope_true"])))
fig.savefig(plot_name)

# ======================================================================================
#
# Have another plot to compare each of the parameters
#
# ======================================================================================
# I'll have several things that need to be tracked for each parameter
params_to_compare = {
    "log_luminosity": "Log Luminosity [e$^-$]",
    "x_pix_snapshot_oversampled": "X Center",
    "y_pix_snapshot_oversampled": "Y Center",
    "scale_radius_pixels": "Scale Radius [pixels]",
    "axis_ratio": "Axis Ratio",
    "position_angle": "Position Angle",
    "power_law_slope": "$\eta$ (Power Law Slope)",
}
param_limits = {
    "log_luminosity": (1, 8),
    "x_pix_snapshot_oversampled": (25, 35),
    "y_pix_snapshot_oversampled": (25, 35),
    "scale_radius_pixels": (0.05, 20),
    "axis_ratio": (-0.05, 1.05),
    "position_angle": (0, np.pi),
    "power_law_slope": (0, 3),
}
param_scale = {
    "log_luminosity": "linear",
    "x_pix_snapshot_oversampled": "linear",
    "y_pix_snapshot_oversampled": "linear",
    "scale_radius_pixels": "log",
    "axis_ratio": "linear",
    "position_angle": "linear",
    "power_law_slope": "linear",
}
# add the true x and y data, they're all the same
catalog["x_pix_snapshot_oversampled_true"] = 30
catalog["y_pix_snapshot_oversampled_true"] = 30

# then plot
fig, axs = bpl.subplots(ncols=4, nrows=2, figsize=[24, 12])
axs = axs.flatten()

for p, ax in zip(params_to_compare, axs):
    ax.scatter(
        catalog[p + "_true"],
        catalog[p + "_best"],
        alpha=1,
        c=plot_colors,
    )

    ax.plot([0, 1e10], [0, 1e10], ls=":", c=bpl.almost_black, zorder=0)
    name = params_to_compare[p]
    ax.add_labels(f"True {name}", f"Measured {name}")
    ax.set_xscale(param_scale[p])
    ax.set_yscale(param_scale[p])
    ax.set_limits(*param_limits[p], *param_limits[p])
    ax.equal_scale()
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("Power Law Slope $\eta$")
    cbar.set_ticks(sorted(np.unique(catalog["power_law_slope_true"])))

fig.savefig(plot_name.parent / "artificial_tests_params.pdf")
