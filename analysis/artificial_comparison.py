"""
artificial_comparison.py
Compare the results of the artificial cluster test to the true results
"""

import sys
from pathlib import Path
import numpy as np
from astropy import table
import cmocean
from matplotlib import ticker, colors, cm, gridspec
from matplotlib import pyplot as plt
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
# get the ratio, and its errorbars
reff_ratio = reff / reff_true
catalog["r_eff_ratio_e+"] = catalog["r_eff_pixels_rmax_15pix_e+"] / reff_true
catalog["r_eff_ratio_e-"] = catalog["r_eff_pixels_rmax_15pix_e-"] / reff_true

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
boundaries = np.arange(0.875, 2.65, 0.25)
norm = colors.BoundaryNorm(
    boundaries,
    ncolors=256,
)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
plot_colors = [mappable.to_rgba(eta) for eta in catalog["power_law_slope_true"]]

# figure size is optimized to make the axes line up properly while also using
# equal_scale on the main comparison axis
fig = plt.figure(figsize=[8, 8.35])
gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    wspace=0.1,
    hspace=0,
    height_ratios=[3, 1],
    width_ratios=[20, 1],
    top=0.97,
    right=0.87,
    left=0.12,
    bottom=0.1,
)
# have two axes: one for the comparison, and one for the ratio
ax_c = fig.add_subplot(gs[0, 0], projection="bpl")
ax_r = fig.add_subplot(gs[1, 0], projection="bpl")

mew = 3
good_size = 5
bad_size = 9
for good_fit, symbol, size in zip([True, False], ["o", "x"], [good_size, bad_size]):
    for eta in sorted(np.unique(catalog["power_law_slope_true"])):
        color = mappable.to_rgba(eta)
        # get the clusters that have this eta and fit quality
        eta_mask = catalog["power_law_slope_true"] == eta
        fit_mask = catalog["good_radius"] == good_fit
        mask = np.logical_and(eta_mask, fit_mask)

        ax_c.errorbar(
            reff_true[mask],
            reff[mask],
            yerr=[
                catalog["r_eff_pixels_rmax_15pix_e-"][mask],
                catalog["r_eff_pixels_rmax_15pix_e+"][mask],
            ],
            fmt=symbol,
            alpha=1,
            markersize=size,
            markeredgewidth=mew,
            markeredgecolor=color,
            color=color,
        )
        # only plot the good fits in the ratio plot
        if good_fit:
            ax_r.errorbar(
                reff_true[mask],
                reff_ratio[mask],
                yerr=[
                    catalog["r_eff_ratio_e-"][mask],
                    catalog["r_eff_ratio_e+"][mask],
                ],
                fmt=symbol,
                alpha=1,
                markersize=size,
                markeredgewidth=mew,
                markeredgecolor=color,
                color=color,
            )
# one to one line and horizontal line for ratio of 1
ax_c.plot([1e-5, 100], [1e-5, 100], ls=":", c=bpl.almost_black, zorder=0)
ax_r.axhline(1, ls=":", lw=3)

# fake symbols for legend
ax_c.errorbar(
    [0],
    [0],
    marker="o",
    markersize=good_size,
    markeredgewidth=mew,
    c=bpl.almost_black,
    label="Success",
)
ax_c.errorbar(
    [0],
    [0],
    marker="x",
    markersize=bad_size,
    markeredgewidth=mew,
    c=bpl.almost_black,
    label="Failure",
)
# plot formatting. Some things common to both axes
for ax in [ax_c, ax_r]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="x", direction="in", which="both")
    ax.tick_params(axis="y", direction="in", which="both")
    ax.xaxis.set_major_formatter(nice_log_formatter)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

# then formatting for each axis separately
x_limits = 0.03, 5
ax_c.set_limits(*x_limits, *x_limits)
ax_c.equal_scale()
ax_c.add_labels("", "Measured $R_{eff}$ [pixels]")
ax_c.yaxis.set_major_formatter(nice_log_formatter)
ax_c.set_xticklabels([])
ax_c.legend(loc=2)

ax_r.set_limits(*x_limits, 1 / 3, 3)
ax_r.add_labels("True $R_{eff}$ [pixels]", "$R_{eff}$ Ratio")
ax_r.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0])
ax_r.set_yticklabels(["", "", "0.5", "", "", "", "", "1", "2", ""])

# the colorbar gets its own axis
cax = fig.add_subplot(gs[:, 1], projection="bpl")
cbar = fig.colorbar(mappable, cax=cax)
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
