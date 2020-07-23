"""
radii_def_comp_plot.py - Compare the two definitions of effective radii

This is done only on the two galaxies used in Ryon+17, as those are the only ones were
we calculate the old version of effective radii.

This script takes the following parameters:
- The path where the plot will be saved
- The path to the output catalogs to include in this plot
"""
import sys
from pathlib import Path
from astropy import table
import betterplotlib as bpl

bpl.set_style()

plot = Path(sys.argv[1])

# ======================================================================================
#
# Load the various catalogs
#
# ======================================================================================
# Go through all the cluster catalogs that are passed in, and get the ones we need
catalogs = {"ngc1313-e": None, "ngc1313-w": None, "ngc628-e": None, "ngc628-c": None}
for item in sys.argv[2:]:
    path = Path(item)
    galaxy_name = path.parent.parent.name
    if galaxy_name in catalogs:
        catalogs[galaxy_name] = table.Table.read(item, format="ascii.ecsv")

# check that none of these are empty
for key in catalogs:
    if catalogs[key] is None:
        raise RuntimeError(f"The {key} catalog has not been created!")

# ======================================================================================
#
# Making plots
#
# ======================================================================================
limits = 0.2, 100
# First we'll make a straight comparison
fig, ax = bpl.subplots(figsize=[7, 7])
for idx, (field, cat) in enumerate(catalogs.items()):
    mask = cat["power_law_slope_best"] > 1.3
    c = bpl.color_cycle[idx]

    ax.scatter(
        cat["r_eff_pc_no_rmax_best"][mask],
        cat["r_eff_pc_rmax_15pix_best"][mask],
        c=c,
        s=5,
        label=field.upper(),
        zorder=2,
        alpha=1,
    )

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_limits(*limits, *limits)
ax.plot(limits, limits, c=bpl.almost_black, lw=1, zorder=0)
ax.equal_scale()
ax.legend(loc=4)
ax.add_labels(
    "Cluster $R_{eff}$ [pc] - Method From Ryon+17",
    "Cluster $R_{eff}$ [pc] - New Method",
)
fig.savefig(plot)
