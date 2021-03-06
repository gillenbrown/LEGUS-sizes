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

# need to add the correct path to import utils
code_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(code_home_dir / "pipeline"))
import utils

bpl.set_style()

# Here I put each galaxy into its own table. This is a bit tricky since some are split
# between fields, and one field is a mosaic with multiple galaxies.
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

# I will have two panels. One will show galaxies with similar radius distributions, the
# other will show ones that deviate. Each panel will have an "other galaxies" category,
# but this "other galaxies" category must be different between the two panels, as the
# sample is different. I have the galaxies manually listed here, but this includes all
# galaxies with more than 180 clusters with well-measured radii. They are also listed
# in order of decreasing cluster number, making them easier to plot
galaxies_1 = ["ngc5194", "ngc628", "ngc1313", "ngc4449"]  # , "ngc3344"]
galaxies_2 = ["ngc1566", "ngc7793"]

individual_cats = dict()
other_cats_1 = []
other_cats_2 = []
all_catalogs = []
for galaxy, cat in galaxy_catalogs.items():
    all_catalogs.append(cat)
    # if it's one of the ones to save, save it
    if galaxy in galaxies_1 or galaxy in galaxies_2:
        individual_cats[cat["galaxy"][0]] = cat
    # separately determine the "other" category
    if galaxy not in galaxies_1 and galaxy not in galaxies_2:
        other_cats_1.append(cat)
    if galaxy not in galaxies_2:
        other_cats_2.append(cat)

big_catalog = table.vstack(all_catalogs, join_type="inner")
other_cat_1 = table.vstack(other_cats_1, join_type="inner")
other_cat_2 = table.vstack(other_cats_2, join_type="inner")


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


# set the colors to be used on the plots
colors = {
    "ngc5194": bpl.color_cycle[0],
    "ngc628": bpl.color_cycle[1],
    "ngc1313": bpl.color_cycle[3],
    "ngc4449": bpl.color_cycle[4],
    "other_1": bpl.color_cycle[5],
    "ngc1566": bpl.color_cycle[7],
    "ngc7793": bpl.color_cycle[6],
    "other_2": bpl.almost_black,
}

fig, axs = bpl.subplots(ncols=2, figsize=[14, 7])
radii_plot = np.logspace(-1, 1.5, 300)
for idx, galaxy in enumerate(galaxies_1 + galaxies_2):
    if galaxy in galaxies_1:
        ax = axs[0]
    else:
        ax = axs[1]

    cat = individual_cats[galaxy]

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
        c=colors[galaxy],
        lw=3,
        label=label,
        zorder=10 - idx,
    )
# plot all other galaxies on both
axs[0].plot(
    radii_plot,
    kde(
        radii_plot,
        other_cat_1["r_eff_log"],
        other_cat_1["r_eff_log_smooth"],
    ),
    lw=3,
    c=colors["other_1"],
    zorder=15,
    label=f"Other Galaxies, N={len(other_cat_1)}",
)
axs[1].plot(
    radii_plot,
    kde(
        radii_plot,
        other_cat_2["r_eff_log"],
        other_cat_2["r_eff_log_smooth"],
    ),
    lw=3,
    c=colors["other_2"],
    zorder=15,
    label=f"All Other Galaxies, N={len(other_cat_2)}",
)


for ax in axs:
    ax.set_xscale("log")
    ax.set_limits(0.1, 20, 0, 1.3)
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(["0.1", "1", "10"])
    ax.add_labels("$R_{eff}$ [pc]", "Normalized KDE Density")
    ax.legend(frameon=False, fontsize=14, loc=2)

# then add all the pixel sizes
# have to get the real directories
data_dir = code_home_dir / "data"
# first set the defaults, then override some key ones. The field doesn't matter if there
# are multiple, since the pixel size will be the same for both.
galaxy_dirs = {galaxy: data_dir / galaxy for galaxy in galaxies_1 + galaxies_2}
galaxy_dirs["ngc5194"] = data_dir / "ngc5194-ngc5195-mosaic"
galaxy_dirs["ngc628"] = data_dir / "ngc628-c"
galaxy_dirs["ngc1313"] = data_dir / "ngc1313-e"
galaxy_dirs["ngc7793"] = data_dir / "ngc7793-e"
# then do what we need to do
for galaxy in galaxies_1 + galaxies_2:
    if galaxy in galaxies_1:
        ax = axs[0]
    else:
        ax = axs[1]

    pixel_size_arcsec = utils.pixels_to_arcsec(1, galaxy_dirs[galaxy])
    pixel_size_pc = utils.arcsec_to_pc_with_errors(
        galaxy_dirs[galaxy], pixel_size_arcsec, 0, 0, False
    )[0]
    ax.plot(
        [pixel_size_pc, pixel_size_pc],
        [0, 0.07],
        lw=3,
        c=colors[galaxy],
        zorder=0,
    )

fig.savefig(plot_name)
