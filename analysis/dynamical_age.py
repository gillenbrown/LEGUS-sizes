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

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1]).resolve()

catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[2:]]
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# throw out some clusters. - same ones as MRR does
mask = np.logical_and.reduce(
    [
        big_catalog["good"],
        big_catalog["mass_msun"] > 0,
        big_catalog["mass_msun_max"] > 0,
        big_catalog["mass_msun_min"] > 0,
        big_catalog["mass_msun_min"] <= big_catalog["mass_msun"],
        big_catalog["mass_msun_max"] >= big_catalog["mass_msun"],
        big_catalog["Q_probability"] > 1e-3,
    ]
)
big_catalog = big_catalog[mask]

# then calculate the dynamical age
big_catalog["dynamical_age"] = big_catalog["age_yr"] / big_catalog["crossing_time_yr"]
print("below", np.sum(big_catalog["dynamical_age"] < 1))
print("above", np.sum(big_catalog["dynamical_age"] > 1))

# ======================================================================================
#
# make the simple plot
#
# ======================================================================================
fig, ax = bpl.subplots()
ax.scatter(big_catalog["mass_msun"], big_catalog["dynamical_age"], s=5, alpha=1)
ax.add_labels("Mass [$M_\odot$]", "Dynamical Age = Age / Crossing Time")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_limits(1e2, 1e6, 1e-3, 1e5)
ax.axhline(1, ls="--")
fig.savefig(plot_name)
