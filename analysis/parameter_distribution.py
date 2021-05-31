"""
parameter_distribution.py

Plot the distribution of the fitting parameters for the clusters in the whole sample,
all on one plot.
"""
import sys

import numpy as np
from astropy import table

import betterplotlib as bpl

bpl.set_style()

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
plot_name = sys.argv[1]
big_catalog = table.Table.read(sys.argv[2], format="ascii.ecsv")

# ======================================================================================
#
# Then make the plot
#
# ======================================================================================
fig, axs = bpl.subplots(ncols=3, nrows=3, figsize=[15, 11])
axs = axs.flatten()

axs[0].hist(big_catalog["log_luminosity"], bins=np.linspace(1, 10, 41))
axs[0].add_labels("Log Luminosity [e$^-$]", "Number of Clusters")
axs[0].set_limits(1, 10)

axs[1].hist(big_catalog["x_fitted"] - big_catalog["x_pix_single"], bin_size=0.1)
axs[1].add_labels("X Position from LEGUS [pixels]", "Number of Clusters")

axs[2].hist(big_catalog["y_fitted"] - big_catalog["y_pix_single"], bin_size=0.1)
axs[2].add_labels("Y Position from LEGUS [pixels]", "Number of Clusters")

x_min, x_max = 0.03, 200
axs[3].hist(
    big_catalog["scale_radius_pixels"],
    bins=np.logspace(np.log10(x_min), np.log10(x_max), 41),
)
axs[3].add_labels("Scale Radius [pixels]", "Number of Clusters")
axs[3].set_limits(x_min, x_max)
axs[3].set_xscale("log")
axs[3].axvline(0.1, ls=":")
axs[3].axvline(15, ls=":")

axs[4].hist(big_catalog["axis_ratio"], bin_size=0.05)
axs[4].add_labels("Axis ratio", "Number of Clusters")
axs[4].set_limits(0, 1.05)

axs[5].hist(big_catalog["position_angle"] % np.pi, bins=np.linspace(0, np.pi, 21))
axs[5].add_labels("Position Angle", "Number of Clusters")
axs[5].set_limits(0, np.pi)

axs[6].hist(big_catalog["power_law_slope"], bins=np.arange(0, 10, 0.5))
axs[6].add_labels("$\eta$ (Power Law Slope)", "Number of Clusters")
axs[6].set_limits(0, 10)
axs[6].axvline(1, ls=":")

axs[7].hist(big_catalog["local_background"], bin_size=100)
axs[7].add_labels("Local Background", "Number of Clusters")
axs[7].set_limits(-100, 1000)

axs[8].hist(big_catalog["num_boostrapping_iterations"], bin_size=5)
axs[8].add_labels("Number of Bootstrapping Iterations", "Number of Clusters")
axs[8].set_limits(0)

fig.savefig(plot_name)
