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

plot_name = Path(sys.argv[1])
long_catalogs = dict()
short_catalogs = []
all_catalogs = []
for item in sys.argv[2:]:
    path = Path(item)
    galaxy_name = path.parent.parent.name.replace("-mosaic", "")
    cat = table.Table.read(item, format="ascii.ecsv")
    # add the min and max allowed radii
    cat["r_eff_min"] = cat["r_eff_pc_rmax_15pix_best"] - cat["r_eff_pc_rmax_15pix_e-"]
    cat["r_eff_max"] = cat["r_eff_pc_rmax_15pix_best"] + cat["r_eff_pc_rmax_15pix_e+"]

    # restrict to only clusters with good radii
    cat = cat[cat["good_radius"]]

    if len(cat) > 200:
        long_catalogs[galaxy_name] = cat
    else:
        short_catalogs.append(cat)
    all_catalogs.append(cat)

big_catalog = table.vstack(all_catalogs, join_type="inner")
all_short_catalog = table.vstack(short_catalogs, join_type="inner")


def error_hist(x_min, x_max):
    xs = np.logspace(-2, 2, 1000)
    ys = []
    for x in xs:
        x_above = x > x_min
        x_below = x < x_max
        x_good = np.logical_and(x_above, x_below)
        ys.append(np.sum(x_good))

    # # normalize the y value
    ys = np.array(ys)
    # ys = ys / np.sum(ys)
    return xs, ys


fig, ax = bpl.subplots()

for galaxy in long_catalogs:
    cat = long_catalogs[galaxy]
    ax.plot(
        *error_hist(cat["r_eff_min"], cat["r_eff_max"]),
        lw=2,
        label=f"{galaxy.upper()}, N={len(cat)}",
    )
ax.plot(
    *error_hist(
        all_short_catalog["r_eff_min"],
        all_short_catalog["r_eff_max"],
    ),
    lw=2,
    c=bpl.color_cycle[6],
    label=f"All Other Fields, N={len(all_short_catalog)}",
)
# ax.plot(
#     *error_hist(
#         big_catalog["r_eff_min"][big_catalog["good"]],
#         big_catalog["r_eff_max"][big_catalog["good"]],
#     ),
#     lw=2,
#     c=bpl.almost_black,
#     label=f"Total, N={np.sum(big_catalog['good'])}",
# )

ax.set_xscale("log")
ax.set_limits(0.05, 30, 0)
ax.add_labels("$R_{eff}$ [pc]", "Clusters")
ax.legend(frameon=False, fontsize=14)
fig.savefig(plot_name)
