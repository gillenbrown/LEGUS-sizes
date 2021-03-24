"""
artificial_cluster_catalog.py
Creates a catalog of artificial clusters with known parameters that will be used to
create an image of artificial clusters
"""

import sys
from pathlib import Path
from astropy import table
from scipy import spatial
import numpy as np

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
catalog_name = Path(sys.argv[1]).resolve()
field = sys.argv[2]
# find the cluster catalog for the field of interest
for cat_loc in sys.argv[3:]:
    if field == Path(cat_loc).parent.parent.name:
        cat_observed = table.Table.read(cat_loc, format="ascii.ecsv")
        break

# create the empty catalog we'll fill later
catalog = table.Table([], names=[])
# ======================================================================================
#
# first create the parameters for clusters. I'll add x-y later
#
# ======================================================================================
# Here is how I'll pick the parameters for my fake clusters:
# log_luminosity - will be in the typical range of clusters in this image
# scale_radius_pixels - will change to simulate moving in distance
# axis_ratio - will be a fixed value typical of clusters
# position_angle - will be randomly chosen for each cluster
# power_law_slope - will be iterated over
# I'll create a slightly nonuniform grid of a-eta-L that I'm iterating over. I'll do
# this in a way that allows for each cluster to have a unique effective radius, to make
# plots easier to see. Instead of having the same a-eta and varying L, I'll slightly
# change a each time. I have a diagram in my notebook, page 137
n_eta = 8
n_l = 8
n_a = 8
eta_values = np.linspace(1.001, 2.5, n_eta)
log_luminosity_values = np.linspace(
    np.min(cat_observed["log_luminosity_best"]),
    np.max(cat_observed["log_luminosity_best"]),
    n_l,
)
a_values = np.logspace(-2, 0, n_l * n_a)

a_final, eta_final, l_final = [], [], []
for a_idx in range(len(a_values)):
    a = a_values[a_idx]
    l = log_luminosity_values[a_idx % n_l]

    for eta in eta_values:
        a_final.append(a)
        eta_final.append(eta)
        l_final.append(l)

# double check my lengths of these arrays
assert len(eta_final) == len(a_final) == len(l_final) == n_eta * n_l * n_a


# then add this all to the table, including IDs
catalog["ID"] = range(1, len(a_final) + 1)
catalog["log_luminosity_true"] = l_final
catalog["scale_radius_pixels_true"] = a_final
catalog["axis_ratio_true"] = 0.8
catalog["position_angle_true"] = np.random.uniform(0, np.pi, len(a_final))
catalog["power_law_slope_true"] = eta_final

# ======================================================================================
#
# Create the x-y positions of the fake clusters
#
# ======================================================================================
# to select x-y values, I have a few rules. First, clusters must be in the region where
# real clusters are (i.e. no edge of the image). They must also not be near other
# clusters, which I define as being outside of 30 pixels from them.
x_real = cat_observed["x"]
y_real = cat_observed["y"]
# Use these to create a region such that we can test whether proposed clusters lie
# within it. The idea is to use a convex hull, but this stack overflow does it a bit
# differently in scipy: https://stackoverflow.com/a/16898636
class Hull(object):
    def __init__(self, x, y):
        hull_points = np.array([(xi, yi) for xi, yi in zip(x, y)])
        self.hull = spatial.Delaunay(hull_points)

    def test_within(self, x, y):
        return self.hull.find_simplex((x, y)) >= 0


hull = Hull(x_real, y_real)

# also get the range so I can restrict where I sample from
max_x_real = np.max(x_real)
max_y_real = np.max(y_real)
min_diff = 30

x_fake, y_fake = np.array([]), np.array([])
for _ in range(len(catalog)):
    # set a counter to track when we have a good set of xy
    good_xy = False
    tracker = 0
    while not good_xy:
        # generate a set of xy
        x = np.random.uniform(0, max_x_real)
        y = np.random.uniform(0, max_y_real)
        within_range = hull.test_within(x, y)

        # then test it against other clusters. The proposed location must be far from
        # every other cluster in either x or y
        far_x_real = np.abs(x_real - x) > min_diff
        far_y_real = np.abs(y_real - y) > min_diff
        far_real = np.all(np.logical_or(far_x_real, far_y_real))
        # and clusters that have been made so far
        far_x_fake = np.abs(x_fake - x) > min_diff
        far_y_fake = np.abs(y_fake - y) > min_diff
        far_fake = np.all(np.logical_or(far_x_fake, far_y_fake))

        far_all = np.logical_and(far_real, far_fake)

        good_xy = np.logical_and(far_all, within_range)

        # make sure we never have an infinite loop
        tracker += 1
        if tracker > 100:
            raise RuntimeError("It appears we can't place any more clusters.")

    x_fake = np.append(x_fake, x)
    y_fake = np.append(y_fake, y)


catalog["x"] = x_fake
catalog["y"] = y_fake

# ======================================================================================
#
# Then add a few other needed parameters before saving the catalog
#
# ======================================================================================
# My pipeline uses these quantities for later analysis, even if I won't ever look at
# the results of this analysis for these artificial clusters.
catalog["mass_msun"] = 1e4
catalog["mass_msun_min"] = 1e4
catalog["mass_msun_max"] = 1e4
catalog["age_yr"] = 1e7
catalog["Q_probability"] = 1

catalog.write(catalog_name, format="ascii.ecsv")
