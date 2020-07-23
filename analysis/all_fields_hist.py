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
catalogs = dict()
for item in sys.argv[2:]:
    path = Path(item)
    galaxy_name = path.parent.parent.name
    catalogs[galaxy_name] = table.Table.read(item, format="ascii.ecsv")

# add the min and max allowed radii
for cat in catalogs.values():
    cat["r_eff_min"] = (
        cat["r_eff_pc_rmax_15pix_best"] - cat["r_eff_pc_rmax_15pix_e-_with_dist"]
    )
    cat["r_eff_max"] = (
        cat["r_eff_pc_rmax_15pix_best"] + cat["r_eff_pc_rmax_15pix_e+_with_dist"]
    )

fig, ax = bpl.subplots()
xs = np.logspace(-1, 2, 1000)
for galaxy in catalogs:
    cat = catalogs[galaxy]
    ys = []
    for x in xs:
        x_above = x > cat["r_eff_min"]
        x_below = x < cat["r_eff_max"]
        x_good = np.logical_and(x_above, x_below)
        ys.append(np.sum(x_good))

    # # normalize the y value
    ys = np.array(ys)
    ys = ys / np.sum(ys)

    ax.plot(xs, ys, c="0.8", lw=1)

ax.set_xscale("log")
fig.savefig(plot_name)
