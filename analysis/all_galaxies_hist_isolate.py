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
from scipy import stats, interpolate
import betterplotlib as bpl

bpl.set_style()

# ======================================================================================
#
# Here I put each galaxy into its own table.
#
# ======================================================================================
# This is a bit tricky since some are split between fields, and one field is a
# mosaic with multiple galaxies.
plot_name = Path(sys.argv[1])
galaxy_catalogs = dict()

for item in sys.argv[2:]:
    cat = table.Table.read(item, format="ascii.ecsv")

    # restrict to the clusters with reliable radii
    cat = cat[cat["good_radius"]]

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

# ======================================================================================
#
# Make the stacked table and it's CDF for use in the KS test later
#
# ======================================================================================
# I individually plot all galaxies, but also have the "normal" population, which is the
# sum of all galaxies other than NGC7793 and NGC1566
normal_cats = [
    cat
    for galaxy, cat in galaxy_catalogs.items()
    if galaxy not in ["ngc7793", "ngc1566", "ngc4395"]
    # if galaxy in ["ngc5194"]
]
stacked_catalog = table.vstack(normal_cats, join_type="inner")

# With this stacked catalog, create the cumulative distribution function, which can
# be used to calculate the KS test for individual galaxies
def make_cumulative_histogram(values):
    sorted_values = np.sort(values)
    ys = np.arange(1, 1 + len(sorted_values), 1)
    assert len(ys) == len(sorted_values)
    return sorted_values, ys / np.max(ys)


r_values, cdf = make_cumulative_histogram(stacked_catalog["r_eff_pc_rmax_15pix_best"])
cdf_func = interpolate.interp1d(
    r_values, cdf, kind="linear", bounds_error=False, fill_value=(0, 1)
)

# ======================================================================================
#
# Make the plot
#
# ======================================================================================
# Sort the individual catalogs by the number of clusters
numbers = []
individual_galaxies = []
for galaxy, cat in galaxy_catalogs.items():
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
    ys = 50 * ys / np.sum(ys)
    return ys


nrows = 4
ncols = 8
fig, axs = bpl.subplots(nrows=nrows, ncols=ncols, figsize=[5 * ncols, 5 * nrows])
axs = axs.flatten()

radii_plot = np.logspace(-1, 1.5, 300)
for idx, galaxy in enumerate(sorted_galaxies):
    ax = axs[idx]
    cat = galaxy_catalogs[galaxy]
    ax.plot(
        radii_plot,
        kde(
            radii_plot,
            stacked_catalog["r_eff_log"],
            stacked_catalog["r_eff_log_smooth"],
        ),
        lw=2,
        zorder=2,
        c=bpl.color_cycle[2],
    )
    ax.plot(
        radii_plot,
        kde(
            radii_plot,
            cat["r_eff_log"],
            cat["r_eff_log_smooth"],
        ),
        lw=4,
        zorder=4,
        c=bpl.color_cycle[0],
    )
    # calculate the KS test value. Compare to our base CDF each time.
    pvalue = stats.ks_1samp(
        cat["r_eff_pc_rmax_15pix_best"],
        cdf_func,
        alternative="two-sided",
    )[1]

    ax.set_xscale("log")
    ax.set_limits(0.1, 25, 0, 1.0)
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.add_labels("$R_{eff}$ [pc]", "Normalized KDE Density")
    ax.easy_add_text(f"{galaxy.upper()}\nN={len(cat)}\nP={pvalue:.3g}", "upper left")
# last axis isn't needed, we only have 31 galaxies
axs[-1].axis("off")

fig.savefig(plot_name)
