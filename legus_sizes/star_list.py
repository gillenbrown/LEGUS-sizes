"""
Makes a list of stars that will be used to make the PSF model.

This script takes the following command-line arguments:
1 - Path to the star list that will be created
"""
# https://photutils.readthedocs.io/en/stable/epsf.html
from pathlib import Path
import sys

from matplotlib.pyplot import show
from astropy.io import fits
from astropy import table
from astropy import stats
from astropy import nddata
import photutils
import numpy as np
from matplotlib import colors
import betterplotlib as bpl

bpl.set_style()
# ======================================================================================
#
# Get the parameters the user passed in
#
# ======================================================================================
# start by getting the output catalog name, which we can use to get the home directory
final_catalog = Path(sys.argv[1]).absolute()
home_dir = final_catalog.parent
# We'll need to get the cluster catalog too
cluster_catalog_path = Path(sys.argv[2]).absolute()

# ======================================================================================
#
# Load the image
#
# ======================================================================================
# then we have to find the image we need. This will be in the drc directory, but the
# exact name is uncertain
galaxy_name = home_dir.name
image_dir = home_dir / f"{galaxy_name}_drc"
# it could be one of two instruments: ACS or UVIS.
for instrument in ["acs", "uvis"]:
    image_name = f"hlsp_legus_hst_{instrument}_{galaxy_name}_f555w_v1_drc.fits"
    try:  # to load the image
        hdu_list = fits.open(image_dir / image_name)
        # if it works break out of this
        break
    except FileNotFoundError:
        continue  # go to next band
else:  # no break, image not found
    raise FileNotFoundError(f"No f555w image found in directory:\n{str(image_dir)}")

# DRC images should have the PRIMARY extension
image_data = hdu_list["PRIMARY"].data

# Then subtract off the median background
_, median, std = stats.sigma_clipped_stats(image_data, sigma=2.0)
image_data -= median

# ======================================================================================
#
# Find the stars in the image
#
# ======================================================================================
# Then we can find peaks in this image. I try a threshold of 100 sigma, but check if
# that wasn't enough, and we'll modify. These values were determined via experimenation
if instrument == "uvis":
    fwhm = 2.2  # pixels
else:
    fwhm = 2.2  # TODO: needs to be updated!
threshold = 100 * std
star_finder = photutils.detection.IRAFStarFinder(
    threshold=threshold, fwhm=fwhm, exclude_border=True
)
peaks_table = star_finder.find_stars(image_data)
while len(peaks_table) < 1000:
    print(len(peaks_table))
    threshold /= 2
    star_finder.threshold = threshold
    peaks_table = star_finder.find_stars(image_data)

print(len(peaks_table))
# Then put the brightest objects first
peaks_table.sort("flux")
peaks_table.reverse()  # put biggest peaks first

# ======================================================================================
#
# Automatically throwing out some stars
#
# ======================================================================================
# I won't bother to exclude stars close to the boundary. The user should be able to
# figure that out themselves when they do the selection
# But I will throw out any peaks that are close to one another
# Set the box size that will be used as the size of the PSF image. This is chosen as the
# size of the Ryon+ 17 fitting region. I'll check for duplicates within this region.
width = 31
one_sided_width = int((width - 1) / 2.0)
# throw out anything within this box. Exclude any stars that have another peak within
# this box. We'll initially say everything is isolated, then exclude ones that we
# find to be not isolated. We'll also check that this is true of the clusters in the
# image
peaks_table["isolated"] = [True] * len(peaks_table)
# We'll write this up as a function, as we'll use this to check both the stars and
# clusters, so don't want to have to resuse the same code
def test_star_isolated(star_x, star_y, all_x, all_y, min_separation):
    """
    Returns whether a given star is isolated.

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
    # see which other objects are away from this one
    isolated_x = seps_x > min_separation
    isolated_y = seps_y > min_separation
    # see where either the x or y is too far away
    isolated = np.logical_or(isolated_x, isolated_y)
    # for the star to be isolated, all these value must be true
    return np.all(isolated)


# read in the table
clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")

# when iterating through the rows, we do need to throw out the star itself when checking
# it against other stars. So this changes the loop a bit
stars_x = peaks_table["xcentroid"].data  # to get as numpy array
stars_y = peaks_table["ycentroid"].data
clusters_x = clusters_table["x_pix_single"]
clusters_y = clusters_table["y_pix_single"]
for idx in range(len(peaks_table)):
    star_x = stars_x[idx]
    star_y = stars_y[idx]

    other_x = np.delete(stars_x, idx)  # returns fresh array, stars_x not modified
    other_y = np.delete(stars_y, idx)

    away_from_stars = test_star_isolated(
        star_x, star_y, other_x, other_y, one_sided_width
    )
    away_from_clusters = test_star_isolated(
        star_x, star_y, clusters_x, clusters_y, one_sided_width
    )

    if not (away_from_stars and away_from_clusters):
        peaks_table["isolated"][idx] = False

print(np.sum(peaks_table["isolated"]))
isolated_table = peaks_table[np.where(peaks_table["isolated"])]

# ======================================================================================
#
# TODO: name this
#
# ======================================================================================
def snapshot(full_image, cen_x, cen_y, width):
    # get the subset of the data first
    # get the central pixel
    cen_x_pix = int(np.floor(cen_x))
    cen_y_pix = int(np.floor(cen_y))
    # we'll select a larger subset around that central pixel, then change the plot
    # limits to be just in the center, so that the object always appears at the center
    buffer_half_width = int(np.ceil(width / 2) + 3)
    min_x_pix = cen_x_pix - buffer_half_width
    max_x_pix = cen_x_pix + buffer_half_width
    min_y_pix = cen_y_pix - buffer_half_width
    max_y_pix = cen_y_pix + buffer_half_width
    # then get this slice of the data
    snapshot_data = full_image[min_y_pix:max_y_pix, min_x_pix:max_x_pix]

    # When showing the plot I want the star to be in the very center. To do this I need
    # to get the values for the border in the new pixel coordinates
    cen_x_new = cen_x - min_x_pix
    cen_y_new = cen_y - min_y_pix
    # then make the plot limits
    min_x_plot = cen_x_new - 0.5 * width
    max_x_plot = cen_x_new + 0.5 * width
    min_y_plot = cen_y_new - 0.5 * width
    max_y_plot = cen_y_new + 0.5 * width

    fig, ax = bpl.subplots(figsize=[6, 5])
    vmin, vmax = np.percentile(snapshot_data, [1, 99])
    # TODO: change linear part of colormap, I think that makes plots look funny
    # TODO: first add the colorbar
    norm = colors.SymLogNorm(vmin=vmin, vmax=vmax * 2, linthresh=1 * std)
    ax.imshow(snapshot_data, norm=norm, cmap=bpl.cm.lapaz)
    ax.set_limits(min_x_plot, max_x_plot, min_y_plot, max_y_plot)
    # ax.set_title(text_base.format(n_shown, n_selected))
    show()


# for row in clusters_table[:10]:
#     snapshot(image_data, row["x_pix_single"], row["y_pix_single"], width)

for row in isolated_table[:10]:
    snapshot(image_data, row["xcentroid"], row["ycentroid"], width)


# We'll then plot all these selected peaks, and have the user select if they're good
# or not.
n_shown = 0
n_selected = 0
text_base = "Number Shown={}, Number Selected={}"

# TODO: throw out peaks too close to clusters
# TODO: redo star selection, I have no guarantee that what I've selected are actually
# stars! I need to know the FWHM somehow. Can use DAOstarfinder:
# https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html
# but that needs to know the FWHM. Currently find_peaks just finds maximum values, they
# don't need to be stars!
# I'll use IRAF star finder, as it reports the FWHM of the object
