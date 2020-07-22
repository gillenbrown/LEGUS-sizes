"""
mass_size.py - plot the mass-size relation for LEGUS clusters
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
plot_name = Path(sys.argv[1])
catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[2:]]
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")
ages = big_catalog["age_yr"]
masses = big_catalog["mass_msun"][ages <= 200e6]
radii = big_catalog["r_eff_pc_rmax_15pix_best"][ages <= 200e6]
# ======================================================================================
#
# make the plot
#
# ======================================================================================
def get_r_percentiles(radii, masses, percentile, d_log_M):
    bins = np.logspace(2, 7, int(5 / d_log_M) + 1)

    bin_centers = []
    radii_percentiles = []
    for idx in range(len(bins) - 1):
        lower = bins[idx]
        upper = bins[idx + 1]

        # then find all clusters in this mass range
        mask_above = masses > lower
        mask_below = masses < upper
        mask_good = np.logical_and(mask_above, mask_below)

        good_radii = radii[mask_good]
        if len(good_radii) > 0:
            radii_percentiles.append(np.percentile(good_radii, percentile))
            # the bin centers will be the mean in log space
            bin_centers.append(10 ** np.mean([np.log10(lower), np.log10(upper)]))

    return bin_centers, radii_percentiles


fig, ax = bpl.subplots()

ax.scatter(masses, radii, alpha=1.0, s=1)
# plot the median and the IQR
d_log_M = 0.5
bin_centers, median = get_r_percentiles(radii, masses, 50, d_log_M)
bin_centers2, lower_bound = get_r_percentiles(radii, masses, 25, d_log_M)
bin_centers3, upper_bound = get_r_percentiles(radii, masses, 75, d_log_M)
assert np.array_equal(bin_centers, bin_centers2)
assert np.array_equal(bin_centers, bin_centers3)

ax.plot(bin_centers, median, c=bpl.almost_black, lw=4)
ax.fill_between(x=bin_centers, y1=lower_bound, y2=upper_bound, color="0.8", zorder=0)


ax.set_xscale("log")
ax.set_yscale("log")
ax.set_limits(1e2, 1e6, 0.2, 40)
ax.add_labels("Cluster Mass [M$_\odot$]", "Cluster Effective Radius [pc]")
fig.savefig(plot_name)
