"""
Makes a list of stars that will be inspected by the user for creation of the PSF.

This script takes the following command-line arguments:
1 - Path to the star list that will be created
2 - Path to the pre-existing cluster list. We use this to check which stars are near
    existing clusters
"""
# https://photutils.readthedocs.io/en/stable/epsf.html
from pathlib import Path
import sys

from astropy.io import fits
from astropy import table
from astropy import stats
import photutils
import numpy as np

import utils

# ======================================================================================
#
# Get the parameters the user passed in
#
# ======================================================================================
# start by getting the output catalog name, which we can use to get the home directory
final_catalog = Path(sys.argv[1]).absolute()
home_dir = final_catalog.parent.parent
# We'll need to get the cluster catalog too
cluster_catalog_path = Path(sys.argv[2]).absolute()
width = int(sys.argv[3])

# ======================================================================================
#
# Load the image
#
# ======================================================================================
image_data, instrument = utils.get_f555w_drc_image(home_dir)

# get the noise_level, which will be used later
_, _, noise = stats.sigma_clipped_stats(image_data, sigma=2.0)

# ======================================================================================
#
# Find the stars in the image
#
# ======================================================================================
# Then we can find peaks in this image. I try a threshold of 100 sigma, but check if
# that wasn't enough, and we'll modify. These values were determined via experimentation
if instrument == "uvis":
    fwhm = 1.85  # pixels
else:
    fwhm = 2.2  # TODO: needs to be updated!
threshold = 100 * noise
star_finder = photutils.detection.IRAFStarFinder(
    threshold=threshold, fwhm=fwhm, exclude_border=True
)
peaks_table = star_finder.find_stars(image_data)
while len(peaks_table) < 1000:
    threshold /= 2
    star_finder.threshold = threshold
    peaks_table = star_finder.find_stars(image_data)

# Then put the brightest objects first
peaks_table.sort("peak")
peaks_table.reverse()  # put biggest peaks first
# then add a column for brightness rank
peaks_table["peak_rank"] = range(1, len(peaks_table) + 1)
# delete a few columns that are calculated poorly or useless
del peaks_table["id"]
del peaks_table["sky"]
del peaks_table["flux"]
del peaks_table["mag"]

# ======================================================================================
#
# Identifying troublesome stars
#
# ======================================================================================
# I want to identify stars that may be problematic, because they're near another star or
# because they're a cluster.
one_sided_width = int((width - 1) / 2.0)
# get duplicates within this box. We initially say that everything has nothing near it,
# then will modify that as needed. We also track if something is close enough to a
# cluster to actually be one.
peaks_table["near_star"] = False
peaks_table["near_cluster"] = False
peaks_table["is_cluster"] = False
# We'll write this up as a function, as we'll use this to check both the stars and
# clusters, so don't want to have to reuse the same code
def test_star_near(star_x, star_y, all_x, all_y, min_separation):
    """
    Returns whether a given star is near other objects

    :param star_x: X pixel coordinate of this star
    :param star_y: Y pixel coordinate of this star
    :param all_x: numpy array of x coordinates of objects to check against
    :param all_y: numpy array of y coordinates of objects to check against
    :param min_separation: The minimum separation allowed for two objects to be
                           considered isolated from each other, in pixels. This check
                           is done within a box of 2*min_separation + 1 pixels, not a
                           circle with radius min_separation.
    :return: True if no objects are close to the star
    """
    seps_x = np.abs(all_x - star_x)
    seps_y = np.abs(all_y - star_y)
    # see which other objects are near this one
    near_x = seps_x < min_separation
    near_y = seps_y < min_separation
    # see where both x and y are close
    near = np.logical_and(near_x, near_y)
    # for the star to be near something, any of these can be true
    return np.any(near)


# read in the clusters table
clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")

# when iterating through the rows, we do need to throw out the star itself when checking
# it against other stars. So this changes the loop a bit. We also get the data
# beforehand to reduce accesses
stars_x = peaks_table["xcentroid"].data  # to get as numpy array
stars_y = peaks_table["ycentroid"].data
clusters_x = clusters_table["x_pix_single"]
clusters_y = clusters_table["y_pix_single"]
for idx in range(len(peaks_table)):
    star_x = stars_x[idx]
    star_y = stars_y[idx]

    other_x = np.delete(stars_x, idx)  # returns fresh array, stars_x not modified
    other_y = np.delete(stars_y, idx)

    peaks_table["near_star"][idx] = test_star_near(
        star_x, star_y, other_x, other_y, one_sided_width
    )
    peaks_table["near_cluster"][idx] = test_star_near(
        star_x, star_y, clusters_x, clusters_y, one_sided_width
    )
    peaks_table["is_cluster"][idx] = test_star_near(
        star_x, star_y, clusters_x, clusters_y, 5
    )


# then write the output catalog
peaks_table.write(final_catalog, format="ascii.ecsv")
