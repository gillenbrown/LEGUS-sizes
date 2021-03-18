"""
artificial_cluster_catalog.py
Creates a catalog of artificial clusters with known parameters that will be used to
create an image of artificial clusters
"""

import sys
from pathlib import Path
from astropy import table
import numpy as np

# ======================================================================================
#
# Load the catalogs that were passed in
#
# ======================================================================================
catalog_name = Path(sys.argv[1]).resolve()

# catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[2:]]
# # then stack them together in one master catalog
# big_catalog = table.vstack(catalogs, join_type="inner")
#
# # restrict to clusters with good radii
# big_catalog = big_catalog[big_catalog["good_radius"]]

# ======================================================================================
#
# Select a few clusters to create
#
# ======================================================================================
# Here is how I'll pick the parameters for my fake clusters:
# x - will be placed throughout the image
# y - will be placed throughout the image
# log_luminosity - will be a fixed value typical of clusters
# scale_radius_pixels - will change to simulate moving in distance
# axis_ratio - will be a fixed value typical of clusters
# position_angle - will be randomly chosen for each cluster
# power_law_slope - will be a fixed value typical of clusters
# local_background - this applies to the image, and will be a representative value
log_luminosity = [6, 6, 6, 6, 6, 6, 6]
axis_ratio = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
power_law_slope = [0.75, 1.01, 1.25, 1.5, 1.75, 2, 2.5]
position_angle = 0
background = 50
# double check my lengths of these arrays
assert len(log_luminosity) == len(axis_ratio) == len(power_law_slope)

# ======================================================================================
#
# Then make the catalog with these parameters
#
# ======================================================================================
colnames = [
    "ID",
    "x",
    "y",
    "log_luminosity_true",
    "scale_radius_pixels_true",
    "axis_ratio_true",
    "position_angle_true",
    "power_law_slope_true",
    "local_background_true",
    "mass_msun",
    "mass_msun_min",
    "mass_msun_max",
    "age_yr",
    "Q_probability",
]
catalog = table.Table(
    [[]] * len(colnames),
    names=colnames,
    dtype=[int] + [float] * (len(colnames) - 1),
)

# set up the grid we'll use for scale radius
a_pixels = np.logspace(-1, 1, 20)
# iterate through the other cluster parameters
dx = 60
id = 0
for i in range(len(log_luminosity)):
    x = (id // len(a_pixels)) * dx + dx
    # then iterate through the scale radius
    for a in a_pixels:
        y = (id % len(a_pixels)) * dx + dx
        # calculate the true effective radius

        catalog.add_row(
            [
                id + 1,  # so first cluster has id=1
                x,
                y,
                log_luminosity[i],
                a,
                axis_ratio[i],
                position_angle,
                power_law_slope[i],
                background,
                1e4,  # mass
                1e3,  # mass min
                1e5,  # mass max
                1e8,  # age
                1,  # Q probability
            ]
        )
        id += 1
# TODO: check on non-integer pixel values

# ======================================================================================
#
# Write the catalog
#
# ======================================================================================
catalog.write(catalog_name, format="ascii.ecsv")
