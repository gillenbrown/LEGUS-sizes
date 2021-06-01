from astropy import table
import matplotlib.pyplot as plt
import numpy as np

# ignore
import sys
import betterplotlib as bpl

bpl.set_style()
plot_name = sys.argv[1]
cat_loc_replace = sys.argv[2]
# fmt: off
# end ignore
catalog = table.Table.read(
    cat_loc_replace,
    format="ascii.ecsv"
)

# parse the LEGUS mass errors
catalog["mass_e-"] = catalog["mass_msun"] - catalog["mass_msun_min"]
catalog["mass_e+"] = catalog["mass_msun_max"] - catalog["mass_msun"]

# get the clusters with reliable radii and masses.
mask = catalog["reliable_radius"] & catalog["reliable_mass"]
subset = catalog[mask]

# plot the data
fig, ax = plt.subplots()
ax.errorbar(
    x=subset["mass_msun"],
    y=subset["r_eff_pc"],
    fmt="o",
    markersize=2,
    lw=0.3,
    xerr=[subset["mass_e-"], subset["mass_e+"]],
    yerr=[subset["r_eff_pc_e-"], subset["r_eff_pc_e+"]],
)

# plot formatting
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e6)
ax.set_ylim(0.1, 40)
ax.set_xlabel("Mass [$M_\odot$]")
ax.set_ylabel("Radius [pc]")
# ignore
fig.savefig(plot_name, bbox_inches="tight")
