"""
experiments.py - script that holds temporary experimental things

This is set up this way so that I can easily pass in all the catalogs
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits
from scipy import special, optimize
from matplotlib import colors, cm, ticker
import betterplotlib as bpl
import cmocean
from tqdm import tqdm

# need to add the correct path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
import utils, fit_utils

bpl.set_style()

# ======================================================================================
#
# load the parameters the user passed in
#
# ======================================================================================
sentinel_name = Path(sys.argv[1])
catalogs = []
for item in sys.argv[2:]:
    cat = table.Table.read(item, format="ascii.ecsv")
    cat["galaxy"] = Path(item).parent.parent.name
    catalogs.append(cat)
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# ======================================================================================
#
# Experiments start here
#
# ======================================================================================
mask = big_catalog["good"]

print(f"Clusters with good fits: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["mass_msun"] > 0)
mask = np.logical_and(mask, big_catalog["mass_msun_max"] > 0)
mask = np.logical_and(mask, big_catalog["mass_msun_min"] > 0)
print(f"Clusters with good masses: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["age_yr"] > 0)
mask = np.logical_and(mask, big_catalog["age_yr_min"] > 0)
mask = np.logical_and(mask, big_catalog["age_yr_max"] > 0)
print(f"Clusters with good ages: {np.sum(mask)}")

age_legus = big_catalog["age_yr"][mask]
mass_legus = big_catalog["mass_msun"][mask]
# mass errors are reported as min and max values
mass_err_hi_legus = big_catalog["mass_msun_max"][mask] - mass_legus
mass_err_lo_legus = mass_legus - big_catalog["mass_msun_min"][mask]

r_eff_legus = big_catalog["r_eff_pc_rmax_15pix_best"][mask]
r_eff_err_hi_legus = big_catalog["r_eff_pc_rmax_15pix_e+"][mask]
r_eff_err_lo_legus = big_catalog["r_eff_pc_rmax_15pix_e-"][mask]

# age errors are reported as min and max values
age_err_hi_legus = big_catalog["age_yr_max"][mask] - age_legus
age_err_lo_legus = age_legus - big_catalog["age_yr_min"][mask]

fig, ax = bpl.subplots(figsize=[8, 8])
ax.scatter(mass_legus, age_legus, alpha=1.0, s=3, zorder=4, c=bpl.color_cycle[0])

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_limits(10, 1e7, 5e5, 20e9)
ax.add_labels("Cluster Mass [M$_\odot$]", "Cluster Age [yr]")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.equal_scale()
fig.savefig(Path.home() / "Desktop" / "age_mass.png", bbox_inches="tight")

# # ======================================================================================
#
# Do not remove this line!!
#
# ======================================================================================
sentinel_name.touch()
