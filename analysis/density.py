"""
density.py - Create a plot showing the density of clusters.

This takes the following parameters:
- Path to save the plot
- Then the paths to all the final catalogs.
"""

import sys
from pathlib import Path
from astropy import table
import numpy as np
from matplotlib import colors
from matplotlib import cm
import betterplotlib as bpl

bpl.set_style()

# import a colormap function from the mass-size relation plots
code_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(code_home_dir / "analysis" / "mass_radius_relation"))
from mass_radius_utils_plotting import create_color_cmap

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1]).resolve()

catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[2:]]
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# restrict to clusters with good masses and radii
good_mask = np.logical_and(big_catalog["good_radius"], big_catalog["good_fit"])
big_catalog = big_catalog[good_mask]


# ======================================================================================
#
# Convenience functions
#
# ======================================================================================
def gaussian(x, mean, variance):
    """
    Normalized Gaussian Function at a given value.

    Is normalized to integrate to 1.

    :param x: value to calculate the Gaussian at
    :param mean: mean value of the Gaussian
    :param variance: Variance of the Gaussian distribution
    :return: log of the likelihood at x
    """
    exp_term = np.exp(-((x - mean) ** 2) / (2 * variance))
    normalization = 1.0 / np.sqrt(2 * np.pi * variance)
    return exp_term * normalization


def kde(x_grid, log_x, log_x_err):
    ys = np.zeros(x_grid.size)
    log_x_grid = np.log10(x_grid)

    for lx, lxe in zip(log_x, log_x_err):
        ys += gaussian(log_x_grid, lx, lxe ** 2)

    # # normalize the y value
    ys = np.array(ys)
    ys = 200 * ys / np.sum(ys)  # arbitrary scaling to look nice
    return ys


def contour(ax, mass, r_eff, color, zorder):
    cmap = create_color_cmap(color, 0.1, 0.8)
    common = {
        "percent_levels": [0.5, 0.95],
        "smoothing": [0.2, 0.2],  # dex
        "bin_size": 0.05,  # dex
        "log": True,
        "cmap": cmap,
    }
    ax.density_contourf(mass, r_eff, alpha=0.2, zorder=zorder, **common)
    ax.density_contour(mass, r_eff, zorder=zorder + 1, **common)


# ======================================================================================
#
# Get the quantities we'll need for the plot
#
# ======================================================================================
density_3d = big_catalog["3d_density"]
density_3d_log_err = big_catalog["3d_density_log_err"]
density_2d = big_catalog["surface_density"]
density_2d_log_err = big_catalog["surface_density_log_err"]

# turn these errors into linear space for plotting
density_3d_err_lo = density_3d - 10 ** (np.log10(density_3d) - density_3d_log_err)
density_3d_err_hi = 10 ** (np.log10(density_3d) + density_3d_log_err) - density_3d

density_2d_err_lo = density_2d - 10 ** (np.log10(density_2d) - density_2d_log_err)
density_2d_err_hi = 10 ** (np.log10(density_2d) + density_2d_log_err) - density_2d

# then mass
mass = big_catalog["mass_msun"]
m_err_lo = big_catalog["mass_msun"] - big_catalog["mass_msun"]
m_err_hi = big_catalog["mass_msun_max"] - big_catalog["mass_msun"]

# also set up the masks for age
age = big_catalog["age_yr"]
mask_young = age < 1e7
mask_med = np.logical_and(age >= 1e7, age < 1e8)
mask_old = np.logical_and(age >= 1e8, age < 1e9)

# ======================================================================================
#
# Make the plot
#
# ======================================================================================
fig, axs = bpl.subplots(figsize=[15, 15], ncols=2, nrows=2)
ax_3_k = axs[0][0]
ax_3_m = axs[1][0]
ax_2_k = axs[0][1]
ax_2_m = axs[1][1]

density_grid = np.logspace(-2, 6, 1000)

for mask, name, color, zorder in zip(
    [mask_young, mask_med, mask_old],
    ["Age: 1-10 Myr", "Age: 10-100 Myr", "Age: 100 Myr - 1 Gyr"],
    [bpl.color_cycle[0], bpl.color_cycle[5], bpl.color_cycle[3]],
    [3, 2, 1],
):
    ax_3_k.plot(
        density_grid,
        kde(density_grid, np.log10(density_3d[mask]), density_3d_log_err[mask]),
        c=color,
    )
    ax_2_k.plot(
        density_grid,
        kde(density_grid, np.log10(density_2d[mask]), density_2d_log_err[mask]),
        c=color,
        label=name,
    )

    contour(ax_3_m, mass[mask], density_3d[mask], color, zorder)
    contour(ax_2_m, mass[mask], density_2d[mask], color, zorder)

    # # for the mass-density plots I'll use my contour scatter function. It takes some kwargs,
    # # but also has nested dictionaries, so we have to do some nested dicts here too.
    # contour = {"log": True}
    # common = {
    #     "percent_levels":  [0.5, 0.75, 0.95],
    #     "smoothing":       [0.1, 0.1],  # dex
    #     "bin_size":        0.05,  # dex
    #     "contour_kwargs":  {"cmap": "Blues", **contour},
    #     "contourf_kwargs": contour,
    #     "fill_cmap":       "Blues",
    # }
    # ax_3_m.contour_scatter(mass, density_3d, **common)
    # ax_2_m.contour_scatter(mass, density_2d, **common)


# format axes
ax_2_k.legend(loc=2, fontsize=16)
for ax in axs[0]:
    ax.set_xscale("log")
    ax.set_limits(0.1, 1e5, 0)
for ax in axs[1]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_limits(1e2, 1e6, 0.1, 1e5)


# add labels to the axes
label_mass = "Mass [$M_\odot$]"
label_kde = "Normalized KDE Density"
label_3d = "Density [$M_\odot$/pc$^3$]"
label_2d = "Surface Density [$M_\odot$/pc$^2$]"
ax_3_k.add_labels(label_3d, label_kde)
ax_3_m.add_labels(label_mass, label_3d)
ax_2_k.add_labels(label_2d, label_kde)
ax_2_m.add_labels(label_mass, label_2d)

fig.savefig(plot_name)
