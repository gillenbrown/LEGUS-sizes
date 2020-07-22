"""
experiments.py - script that holds temporary experimental things

This is set up this way so that I can easily pass in all the catalogs
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
import betterplotlib as bpl

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
big_catalog["fractional_err-"] = (
    big_catalog["r_eff_pc_rmax_15pix_e-_with_dist"]
    / big_catalog["r_eff_pc_rmax_15pix_best"]
)
big_catalog["fractional_err+"] = (
    big_catalog["r_eff_pc_rmax_15pix_e+_with_dist"]
    / big_catalog["r_eff_pc_rmax_15pix_best"]
)

fig, axs = bpl.subplots(ncols=2, figsize=[12, 6])
common = {"histtype": "step", "bin_size": 0.02, "lw": 2}
axs[0].hist(big_catalog["fractional_err-"], **common, label="Error -")
axs[0].hist(big_catalog["fractional_err+"], **common, label="Error +")
axs[0].set_limits(0, 1.1)
axs[0].legend()
axs[0].add_labels("Fractional Error", "Number of Clusters")

axs[1].hist(
    big_catalog["fractional_err-"],
    **common,
    cumulative=True,
    density=True,
    label="Error -"
)
axs[1].hist(
    big_catalog["fractional_err+"],
    **common,
    cumulative=True,
    density=True,
    label="Error +"
)
axs[1].set_limits(0, 1.1, 0, 1)
axs[1].add_labels("Fractional Error", "Cumulative Fraction of Clusters")
fig.savefig("testing/fractional_error_distribution.png")

sentinel_name.touch()
