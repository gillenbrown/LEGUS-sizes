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
for eta in sorted(np.unique(catalog["power_law_slope_true"])):
    eta_mask = catalog["power_law_slope_true"] == eta
    s = ax.errorbar(
        reff_true[eta_mask],
        reff[eta_mask],
        yerr=[
            catalog["r_eff_pixels_rmax_15pix_e-"][eta_mask],
            catalog["r_eff_pixels_rmax_15pix_e+"][eta_mask],
        ],
        alpha=1,
        markersize=9,
        markeredgewidth=0,
        c=mappable.to_rgba(eta),
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
    "scale_radius_pixels": "Scale Radius [pixels]",
    "axis_ratio": "Axis Ratio",
    "position_angle": "Position Angle",
    "power_law_slope": "$\eta$ (Power Law Slope)",
}
param_limits = {
    "scale_radius_pixels": (0.05, 20),
    "axis_ratio": (-0.05, 1.05),
    "position_angle": (0, np.pi),
    "power_law_slope": (0, 3),
}
param_scale = {
    "scale_radius_pixels": "log",
    "axis_ratio": "linear",
    "position_angle": "linear",
    "power_law_slope": "linear",
}

# then plot
fig, axs = bpl.subplots(ncols=2, nrows=2, figsize=[12, 12])
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
