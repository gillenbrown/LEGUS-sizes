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
from matplotlib import colors
from matplotlib import cm
import cmocean
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

# restrict to clusters with good masses and radii
mask = np.logical_and(big_catalog["good_radius"], big_catalog["good_fit"])
big_catalog = big_catalog[mask]

# then calculate the dynamical age
big_catalog["dynamical_age"] = big_catalog["age_yr"] / big_catalog["crossing_time_yr"]
print("bound fraction all", np.sum(big_catalog["dynamical_age"] > 1) / len(big_catalog))
massive_mask = big_catalog["mass_msun"] > 5000
print(
    "bound fraction M > 5000",
    np.sum(big_catalog["dynamical_age"][massive_mask] > 1)
    / len(big_catalog[massive_mask]),
)
old_mask = big_catalog["age_yr"] >= 1e7
print(
    "bound fraction age > 1e7",
    np.sum(big_catalog["dynamical_age"][old_mask] > 1) / len(big_catalog[old_mask]),
)

# ======================================================================================
#
# make the simple plot
#
# ======================================================================================
fig, ax = bpl.subplots()
# make the colormap for masses
cmap = cm.get_cmap("gist_earth_r")
# cmap = cmocean.cm.thermal_r
cmap = cmocean.tools.crop_by_percent(cmap, 20, "min")
norm = colors.LogNorm(vmin=1e3, vmax=1e5)
mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
mass_colors = mappable.to_rgba(big_catalog["mass_msun"])
# perturb the ages slightly for plotting purposes
plot_ages = big_catalog["age_yr"]
plot_ages *= np.random.normal(1, 0.1, len(plot_ages))

# then plot and set some limits
ax.scatter(plot_ages, big_catalog["crossing_time_yr"], s=7, alpha=1, c=mass_colors)
ax.add_labels("Age [yr]", "Crossing time [yr]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_limits(1e5, 2e10, 1e5, 3e8)
cbar = fig.colorbar(mappable, ax=ax, extend="both")
cbar.set_label("Mass [$M_\odot$]")

# add the line for boundedness and label it. Have to use the limits to determine the
# proper rotation of the labels
ax.plot([1, 1e12], [1, 1e12], ls="--", lw=2, c=bpl.almost_black, zorder=0)
frac = 1.2
center = 3e5
shared = {"ha": "center", "va": "center", "rotation": 51, "fontsize": 16}
ax.add_text(x=center * frac, y=center / frac, text="Bound", **shared)
ax.add_text(x=center / frac, y=center * frac, text="Unbound", **shared)

fig.savefig(plot_name)
