"""
make_mask_image.py - Create the mask for a given image.

We mask out bright sources near clusters, but do not consider the whole image to save
time. Here the mask will contain these possible values
-2 - never mask
-1 - always mask
any other value - this pixel should be masked for all clusters other than the one with
                  this ID.
These can be postprocessed to turn this into an actual mask. We do this rather
complicated method so that we can mask other clusters if they're close to a cluster, but
then keep the cluster when we need to fit it later.

This script takes the following parameters:
- Path where the final mask image will be saved
- Path to the sigma image
- Path to the cluster catalog
- Size of the snapshots to be used for fitting
"""
import sys
from pathlib import Path

from astropy import table
from astropy.io import fits
import numpy as np
import photutils
from tqdm import tqdm

import utils

# ======================================================================================
#
# Get the parameters the user passed in, load images and catalogs
#
# ======================================================================================
mask_image_path = Path(sys.argv[1]).absolute()
cluster_catalog_path = Path(sys.argv[2]).absolute()
sigma_image_path = Path(sys.argv[3]).absolute()

snapshot_size = 60  # just to be safe, have a large radius where we mask
cluster_mask_radius = 6
star_radius_fwhm_multiplier = 2  # We mask pixels that are within this*FWHM of a star

galaxy_name = mask_image_path.parent.parent.name
image_data, _, _ = utils.get_drc_image(mask_image_path.parent.parent)

sigma_data = fits.open(sigma_image_path)["PRIMARY"].data
clusters_table = table.Table.read(cluster_catalog_path, format="ascii.ecsv")

# ======================================================================================
#
# Helper functions
#
# ======================================================================================
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_stars(data_snapshot, uncertainty_snapshot):
    """
    Create mask image

    This wil use IRAFStarFinder from photutils to find stars in the snapshot. Anything
    more than 5 pixels from the center will be masked, with a radius of FWHM. This
    makes the full width masked twice the FWHM.

    :param data_snapshot: Snapshot to be used to identify sources
    :param uncertainty_snapshot: Snapshow showing the uncertainty.
    :return: masked image, where values with 1 are good, zero is bad.
    """
    threshold = 5 * np.min(uncertainty_snapshot)
    star_finder = photutils.detection.IRAFStarFinder(
        threshold=threshold + np.min(data_snapshot),
        fwhm=2.0,  # slightly larger than real PSF, to get extended sources
        exclude_border=False,
        sharplo=0.8,
        sharphi=5,
        roundlo=0.0,
        roundhi=0.5,
        minsep_fwhm=1.0,
    )
    peaks_table = star_finder.find_stars(data_snapshot)

    # this will be None if nothing was found
    if peaks_table is None:
        return table.Table([[], [], []], names=["x", "y", "fwhm"])

    # throw away some stars
    to_remove = []
    for idx in range(len(peaks_table)):
        row = peaks_table[idx]
        if (
            # throw away things with large FWHM - are probably clusters!
            row["fwhm"] * star_radius_fwhm_multiplier > cluster_mask_radius
            or row["peak"] < row["sky"]
            # peak is sky-subtracted. This ^ removes ones that aren't very far above
            # a high sky background. This cut stops substructure in clusters from
            # being masked.
            or row["peak"] < threshold
        ):
            to_remove.append(idx)
    peaks_table.remove_rows(to_remove)

    xs = peaks_table["xcentroid"].data
    ys = peaks_table["ycentroid"].data
    mask_radius = peaks_table["fwhm"].data * star_radius_fwhm_multiplier
    blank = np.ones(xs.shape) * -1

    return table.Table(
        [xs, ys, mask_radius, blank, blank],
        names=["x", "y", "mask_radius", "near_cluster", "is_cluster"],
    )


# ======================================================================================
#
# Creating the mask around the clusters
#
# ======================================================================================
mask_data = np.ones(image_data.shape, dtype=int) * -2
for cluster in tqdm(clusters_table):
    # create the snapshot. We use ceiling to get the integer pixel values as python
    # indexing does not include the final value. So when we calcualte the offset, it
    # naturally gets biased low. Moving the center up fixes that in the easiest way.
    x_cen = int(np.ceil(cluster["x"]))
    y_cen = int(np.ceil(cluster["y"]))

    x_min = x_cen - int(np.ceil(snapshot_size / 2.0))
    x_max = x_cen + int(np.floor(snapshot_size / 2.0))
    y_min = y_cen - int(np.ceil(snapshot_size / 2.0))
    y_max = y_cen + int(np.floor(snapshot_size / 2.0))

    data_snapshot = image_data[y_min:y_max, x_min:x_max].copy()
    error_snapshot = sigma_data[y_min:y_max, x_min:x_max].copy()
    # find the stars
    these_stars = find_stars(data_snapshot, error_snapshot)
    # change the x and y coordinates from the snapshot coords to image coords
    these_stars["x"] += x_min
    these_stars["y"] += y_min

    # figure out which stars are near clusters
    for star in these_stars:
        for c in clusters_table:
            dist_to_cluster = distance(star["x"], star["y"], c["x"], c["y"])
            # if it's close to the cluster, we won't even bother marking this source
            if dist_to_cluster < 2:
                star["is_cluster"] = c["ID"]
            # otherwise, mark stars that will have any overlap with the fit region
            max_dist = star["mask_radius"] + cluster_mask_radius
            if dist_to_cluster < max_dist:
                star["near_cluster"] = c["ID"]

    # then mask the pixels found here, as well as the cluster itself. We do this here
    # so that we don't have to iterate over the whole image
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            # mark the cluster. We only need to check this cluster since we'll go
            # through all of them individually
            if distance(x, y, cluster["x"], cluster["y"]) < cluster_mask_radius:
                # if there is overlap between regions of different clusters, leave it
                # as unmasked.
                if mask_data[y][x] < 0:
                    mask_data[y][x] = cluster["ID"]
                else:
                    mask_data[y][x] = -2

            # then go through each star
            for star in these_stars:
                if star["is_cluster"] >= 0:
                    continue
                # if this pixel is close to the star, we'll need to mask
                if distance(x, y, star["x"], star["y"]) < star["mask_radius"]:
                    # if it's close to a cluster, mark that value
                    if star["near_cluster"] >= 0:
                        # there should be no overlap with something already marked to
                        # belong to one cluster, unless it's the same cluster
                        assert (
                            mask_data[y][x] < 0
                            or mask_data[y][x] == star["near_cluster"]
                        )
                        mask_data[y][x] = star["near_cluster"]
                    # if it's not near a cluster, it could overlap with one that is.
                    # if that's the case, go ahead and mark this pixel. By our
                    # definition of what it means to be near a cluster, this will not
                    # overlap with any pixels in the cluster itself, only in stars that
                    # are near the cluster.
                    else:
                        mask_data[y][x] = -1

# ======================================================================================
#
# Then write this output image
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(mask_data)
new_hdu.writeto(mask_image_path, overwrite=True)
