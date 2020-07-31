"""
make_mask_image.py - Create the mask for a given image.

We mask out bright sources near clusters, but do not consider the whole image to save
time. Here the mask will contain the following values.
0 - pixel value never to be used
1 - pixel always allowed to be used
2 - pixel is close to a cluster. In the fitting, determine whether this value is at the
    center of the fitted region. If so, leave it as a pixel to be fitted. If not, mask
    it, as it indicates a nearby cluster that could contaminate the image.

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
    fwhm = peaks_table["fwhm"].data

    return table.Table([xs, ys, fwhm], names=["x", "y", "fwhm"])


# ======================================================================================
#
# Creating the mask around the clusters
#
# ======================================================================================
mask_data = np.ones(image_data.shape)
for cluster in tqdm(clusters_table):
    # create the snapshot. We use ceiling to get the integer pixel values as python
    # indexing does not include the final value. So when we calcualte the offset, it
    # naturally gets biased low. Moving the center up fixes that in the easiest way.
    x_cen = int(np.ceil(cluster["x_pix_single"]))
    y_cen = int(np.ceil(cluster["y_pix_single"]))

    x_min = x_cen - int(np.ceil(snapshot_size / 2.0))
    x_max = x_cen + int(np.floor(snapshot_size / 2.0))
    y_min = y_cen - int(np.ceil(snapshot_size / 2.0))
    y_max = y_cen + int(np.floor(snapshot_size / 2.0))

    data_snapshot = image_data[y_min:y_max, x_min:x_max]
    error_snapshot = sigma_data[y_min:y_max, x_min:x_max]
    # create the mask
    these_stars = find_stars(data_snapshot, error_snapshot)
    # change the x and y coordinates from the snapshot coords to image coords
    these_stars["x"] += x_min
    these_stars["y"] += y_min

    # then mask the pixels found here, as well as the cluster itself
    for x_idx in range(x_min, x_max):
        x = x_idx + 0.5  # to get pixel center
        for y_idx in range(y_min, y_max):
            y = y_idx + 0.5  # to get pixel center
            dist_cluster = distance(
                x, y, cluster["x_pix_single"], cluster["y_pix_single"]
            )
            # mark the cluster
            if dist_cluster < cluster_mask_radius:
                mask_data[y_idx][x_idx] = 2
            # then go through each star and see if it's close
            for star in these_stars:
                dist_star = distance(x, y, star["x"], star["y"])
                # don't mark pixels that have already been marked as clusters
                if (
                    dist_star < star_radius_fwhm_multiplier * star["fwhm"]
                    and mask_data[y_idx][x_idx] < 1.5
                ):
                    mask_data[y_idx][x_idx] = 0


# ======================================================================================
#
# Then write this output image
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(mask_data)
new_hdu.writeto(mask_image_path, overwrite=True)
