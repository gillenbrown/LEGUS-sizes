"""
parameter_distribution.py

Plot the distribution of the fitting parameters for the clusters in the whole sample,
all on one plot.
"""
import sys

from astropy import table

import betterplotlib as bpl

bpl.set_style()

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = sys.argv[1]
catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[2:]]
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# ======================================================================================
#
# Then make the plot
#
# ======================================================================================
fig, axs = bpl.subplots(ncols=2, nrows=2, figsize=[10, 10])
axs = axs.flatten()

axs[0].hist(big_catalog["scale_radius_pixels_median"], bin_size=1)
axs[0].add_labels("Scale Radius [pixels]", "Number of Clusters")
axs[0].set_limits(0)

axs[1].hist(big_catalog["axis_ratio_median"], bin_size=0.05)
axs[1].add_labels("Axis ratio", "Number of Clusters")
axs[1].set_limits(0, 1)

axs[2].hist(big_catalog["power_law_slope_median"], bin_size=0.2)
axs[2].add_labels("$\eta$ (Power Law Slope)", "Number of Clusters")
axs[2].set_limits(0, 10)
axs[2].axvline(1, ls=":")

axs[3].hist(big_catalog["num_boostrapping_iterations"], bin_size=5)
axs[3].add_labels("Number of Bootstrapping Iterations", "Number of Clusters")
axs[3].set_limits(0)

fig.savefig(plot_name)
