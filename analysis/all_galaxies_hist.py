"""
all_fields_hist.py - Create a histogram showing the distribution of effective radii in
all fields.

This takes the following parameters:
- Path to save the plots
- Then the paths to all the final catalogs.
"""

import sys
from pathlib import Path
from astropy import table
import numpy as np
import betterplotlib as bpl

bpl.set_style()

# Here I put each galaxy into its own table. This is a bit tricky since some are split
# between fields, and one field is a mosaic with multiple galaxies.
plot_name = Path(sys.argv[1])
galaxy_catalogs = dict()

for item in sys.argv[2:]:
    cat = table.Table.read(item, format="ascii.ecsv")

    # throw out some clusters. - same ones as MRR does
    mask = np.logical_and.reduce(
        [
            cat["good"],
            cat["mass_msun"] > 0,
            cat["mass_msun_max"] > 0,
            cat["mass_msun_min"] > 0,
            cat["mass_msun_min"] <= cat["mass_msun"],
            cat["mass_msun_max"] >= cat["mass_msun"],
            cat["Q_probability"] > 1e-3,
        ]
    )
    cat = cat[mask]

    # calculate the mean error in log space, will be used for the KDE smothing
    r_eff = cat["r_eff_pc_rmax_15pix_best"]
    log_r_eff = np.log10(r_eff)
    log_err_lo = log_r_eff - np.log10(r_eff - cat["r_eff_pc_rmax_15pix_e-"])
    log_err_hi = np.log10(r_eff + cat["r_eff_pc_rmax_15pix_e+"]) - log_r_eff

    cat["r_eff_log"] = log_r_eff
    mean_log_err = 0.5 * (log_err_lo + log_err_hi)
    cat["r_eff_log_smooth"] = 1.75 * mean_log_err  # don't do average, it is too small

    # go through all galaxies in this field
    for galaxy in np.unique(cat["galaxy"]):
        galaxy_table = cat[cat["galaxy"] == galaxy]
        # throw away the part after the hyphen, which will be the field for those split
        galaxy = galaxy.split("-")[0]
        # rename this in the split
        galaxy_table["galaxy"] = galaxy
        # then store this part of the catalog. If one for this galaxy already exists,
        # append it
        if galaxy in galaxy_catalogs:
            galaxy_catalogs[galaxy] = table.vstack(
                [galaxy_catalogs[galaxy], galaxy_table], join_type="inner"
            )
        else:
            galaxy_catalogs[galaxy] = galaxy_table

# Then figure out which galaxies to independently plot, and which to throw in an
# "other" category
long_catalogs = dict()
short_catalogs = []
all_catalogs = []
for galaxy, cat in galaxy_catalogs.items():
    all_catalogs.append(cat)
    if len(cat) > 150:
        long_catalogs[cat["galaxy"][0]] = cat
    else:
        short_catalogs.append(cat)

big_catalog = table.vstack(all_catalogs, join_type="inner")
all_short_catalog = table.vstack(short_catalogs, join_type="inner")

# for those which are plotted individually, sort them by the number of clusters
numbers = []
individual_galaxies = []
for galaxy, cat in long_catalogs.items():
    individual_galaxies.append(galaxy)
    numbers.append(len(cat))

idx_sort = np.argsort(numbers)[::-1]  # largest first
sorted_galaxies = np.array(individual_galaxies)[idx_sort]


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


def kde(r_eff_grid, log_r_eff, log_r_eff_err):
    ys = np.zeros(r_eff_grid.size)
    log_r_eff_grid = np.log10(r_eff_grid)

    for lr, lre in zip(log_r_eff, log_r_eff_err):
        ys += gaussian(log_r_eff_grid, lr, lre ** 2)

    # # normalize the y value
    ys = np.array(ys)
    ys = 70 * ys / np.sum(ys)
    return ys


fig, ax = bpl.subplots(figsize=[8, 7])
radii_plot = np.logspace(-1, 1.5, 300)
for idx, galaxy in enumerate(sorted_galaxies):
    cat = long_catalogs[galaxy]

    # I want to add an extra space in the legend for NGC628
    label = f"NGC {galaxy[3:]}, "
    if len(galaxy) == 6:
        # chose spaces fine tuned to align in the legend:
        # https://www.overleaf.com/learn/latex/Spacing_in_math_mode
        label += "$\  \ $"
    label += f"N={len(cat)}"

    ax.plot(
        radii_plot,
        kde(
            radii_plot,
            cat["r_eff_log"],
            cat["r_eff_log_smooth"],
        ),
        lw=3,
        label=label,
        zorder=10 - idx,
    )
ax.plot(
    radii_plot,
    kde(
        radii_plot,
        all_short_catalog["r_eff_log"],
        all_short_catalog["r_eff_log_smooth"],
    ),
    lw=3,
    c=bpl.color_cycle[7],
    zorder=15,
    label=f"All Other Galaxies, N={len(all_short_catalog)}",
)
ax.plot(
    radii_plot,
    kde(
        radii_plot,
        big_catalog["r_eff_log"],
        big_catalog["r_eff_log_smooth"],
    ),
    lw=5,
    c=bpl.almost_black,
    zorder=20,
    label=f"Total, N={len(big_catalog)}",
)

ax.set_xscale("log")
ax.set_limits(0.1, 25, 0)
ax.set_xticks([0.1, 1, 10])
ax.set_xticklabels(["0.1", "1", "10"])
ax.add_labels("$R_{eff}$ [pc]", "Normalized KDE Density")
ax.legend(frameon=False, fontsize=14)
fig.savefig(plot_name)
