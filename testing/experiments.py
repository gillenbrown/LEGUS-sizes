"""
experiments.py - script that holds temporary experimental things

This is set up this way so that I can easily pass in all the catalogs
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from matplotlib import colors, cm
import betterplotlib as bpl
import cmocean

bpl.set_style()

# ======================================================================================
#
# load the parameters the user passed in
#
# ======================================================================================
sentinel_name = Path(sys.argv[1])
catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[2:]]
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# ======================================================================================
#
# Experiments start here
#
# ======================================================================================
def path_to_no_prior(path_with_prior):
    return path_with_prior.replace("catalog_30_pixels", "catalog_no_priors_30_pixels")


def get_galaxy(cat_path):
    return Path(cat_path).parent.parent.name


cats_prior = dict()
cats_no_prior = dict()
for path in sys.argv[2:]:
    gal = get_galaxy(path)
    cats_prior[gal] = table.Table.read(path, format="ascii.ecsv")
    cats_no_prior[gal] = table.Table.read(path_to_no_prior(path), format="ascii.ecsv")

# then match them
matches = dict()
for gal in cats_prior:
    matches[gal] = table.join(
        cats_prior[gal], cats_no_prior[gal], join_type="inner", keys="ID"
    )
    matches[gal]["galaxy"] = gal  # .ljust(20)
# put these all into a big column for easy use
big_catalog = table.vstack(list(matches.values()), join_type="inner")
# colnames with _1 appended are with priors, with _2 are without priors
colname_reff_with_prior = "r_eff_pixels_rmax_15pix_best_1"
colname_reff_no_prior = "r_eff_pixels_rmax_15pix_best_2"

# find the ones I want to select - I'll color code them separately on the plot
lower_no_prior = 0
upper_no_prior = 0.8
lower_ratio = 0
upper_ratio = 0.6
mask1 = np.logical_and(
    big_catalog[colname_reff_no_prior] > lower_no_prior,
    big_catalog[colname_reff_no_prior] < upper_no_prior,
)
ratio = big_catalog[colname_reff_with_prior] / big_catalog[colname_reff_no_prior]
mask2 = np.logical_and(ratio > lower_ratio, ratio < upper_ratio,)
mask = np.logical_and(mask1, mask2)

eta_no_prior = big_catalog["power_law_slope_best_2"]
eta_with_prior = big_catalog["power_law_slope_best_1"]
eta_ratio = eta_no_prior / eta_with_prior


fig, ax = bpl.subplots()

# cmap = cmocean.cm.thermal_r
# norm = colors.Normalize(vmin=0.4, vmax=5)
# mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
# point_colors = mappable.to_rgba(eta_no_prior)
# axs[0].scatter(
#     big_catalog[colname_reff_no_prior],
#     big_catalog[colname_reff_with_prior],
#     s=2,
#     c=point_colors,
#     alpha=1,
#     zorder=1,
# )
# cbar = fig.colorbar(mappable, ax=axs[0], pad=0)
# cbar.set_label("$\eta$ No Prior")


cmap = cm.coolwarm
norm = colors.LogNorm(vmin=0.5, vmax=2)
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
point_colors = mappable.to_rgba(eta_ratio)
ax.scatter(
    big_catalog[colname_reff_no_prior],
    big_catalog[colname_reff_with_prior],
    s=2,
    c=point_colors,
    alpha=1,
    zorder=1,
)
cbar = fig.colorbar(mappable, ax=ax, pad=0)
cbar.set_label("$\eta$ No Prior / $\eta$ With Prior")
ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]
cbar.set_ticks(ticks)
cbar.set_ticklabels([str(t) for t in ticks])

ax.scatter(
    big_catalog[colname_reff_no_prior][mask],
    big_catalog[colname_reff_with_prior][mask],
    s=30,
    alpha=1,
    zorder=2,
    c=bpl.color_cycle[1],
)
ax.add_labels("$R_{eff}$ without priors [pixels]", "$R_{eff}$ with priors [pixels]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_limits(0.1, 20, 0.1, 20)
ax.plot([0.001, 100], [0.001, 100], c=bpl.almost_black, ls="--", zorder=0, lw=1)
ax.equal_scale()
fig.savefig(Path(__file__).parent / "priors_affect.png", bbox_inches="tight")


for idx in range(len(mask)):
    if mask[idx]:
        print(
            big_catalog["galaxy"][idx],
            big_catalog["ID"][idx],
            big_catalog[colname_reff_no_prior][idx],
            big_catalog[colname_reff_with_prior][idx],
            big_catalog["scale_radius_pixels_best_1"][idx],
            eta_with_prior[idx],
        )
# ======================================================================================
#
# Do not remove this line!!
#
# ======================================================================================
sentinel_name.touch()
