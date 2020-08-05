"""
fit_quality.py - Make a plot investigating the fit quality

This takes the following parameters
- Path where this plot will be saved
- All the completed catalogs
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from matplotlib import pyplot as plt
from matplotlib import gridspec, colors, cm
import cmocean
import colorcet
import betterplotlib as bpl

bpl.set_style()

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1]).resolve()
catalogs = []
for item in sys.argv[2:]:
    cat = table.Table.read(item, format="ascii.ecsv")
    cat["galaxy"] = Path(item).parent.parent.name
    catalogs.append(cat)
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")
# multiply the MAD by 100 to get it to percent
# big_catalog["profile_mad"] *= 100

# ======================================================================================
#
# Plot setup
#
# ======================================================================================
# I'll have several things that need to be tracked for each parameter
params = {
    # "central_surface_brightness_best": "Central Surface Brightness [e$^-$]",
    "scale_radius_pixels_best": "Scale Radius [pixels]",
    "axis_ratio_best": "Axis Ratio",
    # "position_angle_best": "Position Angle",
    "power_law_slope_best": "$\eta$ (Power Law Slope)",
    # "local_background_best": "Local Background [e$^-$]",
}
param_limits = {
    "central_surface_brightness_best": (10, 1e8),
    "scale_radius_pixels_best": (1e-7, 1e4),
    "axis_ratio_best": (-0.05, 1.05),
    "position_angle_best": (0, np.pi),
    "power_law_slope_best": (0, 3),
    "local_background_best": (-500, 1000),
}
param_scale = {
    "central_surface_brightness_best": "log",
    "scale_radius_pixels_best": "log",
    "axis_ratio_best": "linear",
    "position_angle_best": "linear",
    "power_law_slope_best": "linear",
    "local_background_best": "linear",
}
param_bins = {
    "central_surface_brightness_best": np.logspace(1, 8, 41),
    "scale_radius_pixels_best": np.logspace(-7, 4, 41),
    "axis_ratio_best": np.arange(-0.1, 1.1, 0.05),
    "position_angle_best": np.arange(0, 3.5, 0.1),
    "power_law_slope_best": np.arange(0, 3.2, 0.1),
    "local_background_best": np.arange(-300, 1500, 100),
}

indicators = {
    "fit_rms": "Fit RMS",
    "profile_mad": "Relative MAD of\nCumulative Profile",
    "estimated_local_background_diff_sigma": "Estimated Local\nBackground Error",
}
ind_limits = {
    "fit_rms": (0.1, 100),
    "profile_mad": (1e-3, 1),
    "estimated_local_background_diff_sigma": (-3, 3),
}
ind_scale = {
    "fit_rms": "log",
    "profile_mad": "log",
    "estimated_local_background_diff_sigma": "linear",
}


def get_percentiles(xs, ys, percentile, bins, bin_scale):
    bin_centers = []
    ys_percentiles = []

    # throw out nans
    mask = ~np.isnan(ys)
    xs = xs[mask]
    ys = ys[mask]
    for idx in range(len(bins) - 1):
        lower = bins[idx]
        upper = bins[idx + 1]

        # then find all clusters in this mass range
        mask_above = xs > lower
        mask_below = xs <= upper
        mask_good = np.logical_and(mask_above, mask_below)

        good_ys = ys[mask_good]
        if len(good_ys) > 0:
            ys_percentiles.append(np.percentile(good_ys, percentile))
            # the bin centers will be the mean in log space
            if bin_scale == "log":
                bin_center = 10 ** np.mean([np.log10(lower), np.log10(upper)])
            else:
                bin_center = np.mean([lower, upper])
            bin_centers.append(bin_center)

    return bin_centers, ys_percentiles


# ======================================================================================
#
# Then make the plot
#
# ======================================================================================
# This will have several columns for different parameters, with the rows being the
# different ways of assessing each parameter
fig = plt.figure(figsize=[6 * len(params), 4 * (1 + len(indicators))])
gs = gridspec.GridSpec(
    nrows=len(indicators) + 1,
    ncols=len(params),
    wspace=0.2,
    hspace=0.1,
    left=0.1,
    right=0.98,
    bottom=0.06,
    top=0.96,
)

# Then go through and make the columns
for idx_p, param in enumerate(params):
    # add the histogram
    ax = fig.add_subplot(gs[0, idx_p], projection="bpl")
    ax.hist(big_catalog[param], bins=param_bins[param])
    ax.set_title(params[param])
    if idx_p == 0:
        ax.add_labels(y_label="Number of Clusters")
    ax.set_limits(*param_limits[param])
    ax.set_xscale(param_scale[param])

    # then the indicator plots
    for idx_q, indicator in enumerate(indicators, start=1):
        ax = fig.add_subplot(gs[idx_q, idx_p], projection="bpl")

        ax.scatter(big_catalog[param], big_catalog[indicator], alpha=1, s=1, zorder=2)

        # Draw the percentile lines
        for percentile in [5, 25, 50, 75, 95]:
            xs, ys = get_percentiles(
                big_catalog[param],
                big_catalog[indicator],
                percentile,
                param_bins[param],
                param_scale[param],
            )
            ax.plot(
                xs,
                ys,
                c=bpl.almost_black,
                lw=2 * (1 - (abs(percentile - 50) / 50)) + 1,
                zorder=1,
            )
            ax.text(
                x=xs[1], y=ys[1], ha="center", va="bottom", s=percentile, fontsize=16,
            )

        ax.set_limits(*param_limits[param], *ind_limits[indicator])
        # remove the X label and ticks for all but the last plot
        if idx_q == 3:
            # ax.remove_labels("x")
            ax.add_labels(x_label=params[param])
        if idx_p == 0:
            ax.add_labels(y_label=indicators[indicator])
        # set ticks on top and bottom
        ax.tick_params(axis="x", top=True, bottom=True, direction="in")
        ax.set_xscale(param_scale[param])
        ax.set_yscale(ind_scale[indicator])

        if "background" in indicator:
            ax.axhline(0, ls=":", c=bpl.almost_black)


fig.savefig(plot_name)

# ======================================================================================
#
# Then make a plot showing how these parameters relate to one another
#
# ======================================================================================
fig, axs = bpl.subplots(ncols=3, figsize=[20, 6])
axs = axs.flatten()

x_param = "power_law_slope_best"
y_param = "scale_radius_pixels_best"
for ax, color_ind in zip(axs, indicators):

    if color_ind == "profile_mad":
        cmap = cmocean.cm.deep
        norm = colors.LogNorm(vmin=0.005, vmax=0.1)
    elif color_ind == "fit_rms":
        cmap = cmocean.cm.rain
        norm = colors.LogNorm(vmin=0.5, vmax=10)
    else:
        cmap = colorcet.m_bkr
        norm = colors.Normalize(vmin=-1, vmax=1)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    point_colors = [mappable.to_rgba(c) for c in big_catalog[color_ind]]

    ax.scatter(big_catalog[x_param], big_catalog[y_param], c=point_colors, s=2, alpha=1)
    ax.add_labels(params[x_param], params[y_param])
    ax.set_limits(*param_limits[x_param], *param_limits[y_param])
    ax.set_xscale(param_scale[x_param])
    ax.set_yscale(param_scale[y_param])
    cbar = fig.colorbar(mappable, ax=ax, pad=0)
    cbar.set_label(indicators[color_ind])
    ax.axhline(0.1, ls=":")
fig.savefig(plot_name.parent / "param_correlation.png")

# for row in big_catalog:
#     if (
#         row["scale_radius_pixels_best"] < 1e-6
#         # and row["power_law_slope_best"] > 1
#         # and row["profile_mad"] < 1
#     ):
#         print(row["galaxy"], row["ID"])
