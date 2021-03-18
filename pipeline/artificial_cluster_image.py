"""
artificial_cluster_image.py
Make an image full of artificial clusters to test the pipeline with
"""

import sys
from pathlib import Path
from astropy import table
from astropy.io import fits
import numpy as np

import utils, fit_utils

# ======================================================================================
#
# Load the parameters that were passed in
#
# ======================================================================================
image_name = Path(sys.argv[1]).resolve()
true_catalog_name = Path(sys.argv[2]).resolve()
true_catalog = table.Table.read(true_catalog_name, format="ascii.ecsv")

oversampling_factor = int(sys.argv[3])
snapshot_size = 2 * int(sys.argv[4])  # extend further for creation of larger clusters
snapshot_size_oversampled = oversampling_factor * snapshot_size

# load the PSF
psf_path = Path(sys.argv[5]).resolve()
psf = fits.open(psf_path)["PRIMARY"].data
# the convolution requires the psf to be normalized, and without any negative values
psf = np.maximum(psf, 0)
psf /= np.sum(psf)

# then load the image from the suggested galaxy
galaxy = sys.argv[6]
galaxy_dir = image_name.parent.parent / galaxy
base_image = utils._get_image(galaxy_dir)[0]

# ======================================================================================
#
# Then create an array that we'll use to create the image.
#
# ======================================================================================
image = np.zeros(base_image.data.shape)
# I can eventually use the image itself later if I desire

# add the local background. Note that clusters have the same background, which is why
# I can do what I do here.
image += true_catalog["local_background_true"][0]

# ======================================================================================
#
# Then go through and add the artificial clusters to this image
#
# ======================================================================================
for row in true_catalog:
    cluster_snapshot = fit_utils.create_model_image(
        row["log_luminosity_true"],
        snapshot_size,  # x, center of snapshot
        snapshot_size,  # y, center of snapshot
        row["scale_radius_pixels_true"],
        row["axis_ratio_true"],
        row["position_angle_true"],
        row["power_law_slope_true"],
        0,  # background (as it's already in the image)
        psf,
        snapshot_size_oversampled,
        oversampling_factor,
    )[-1]

    # then add this array to the appropriate region of the image. To do this I have
    # to take out the region from the image (to make it match the size of this cluster),
    # then add the artificial cluster to that part of the image. Note that getting the
    # region still allows us to modify the image in place.

    # We use ceiling to get the integer pixel values as python indexing does not
    # include the final value.
    x_cen = int(np.ceil(row["x"]))
    y_cen = int(np.ceil(row["y"]))
    # Get the snapshot, based on the size desired.
    # Since we took the ceil of the center, go more in the negative direction (i.e.
    # use ceil to get the minimum values). This only matters if the snapshot size is
    # odd
    x_min = x_cen - int(np.ceil(snapshot_size / 2.0))
    x_max = x_cen + int(np.floor(snapshot_size / 2.0))
    y_min = y_cen - int(np.ceil(snapshot_size / 2.0))
    y_max = y_cen + int(np.floor(snapshot_size / 2.0))

    image_region = image[y_min:y_max, x_min:x_max]
    image_region += cluster_snapshot

# ======================================================================================
#
# Add noise to the image
#
# ======================================================================================
image += np.random.poisson(image)

# ======================================================================================
#
# write the image
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(image)
# grab the header from the old image
new_hdu.header = base_image.header
# then reset the exposure time to be 1, as I've already put things in electrons.
new_hdu.header["EXPTIME"] = 1
new_hdu.writeto(image_name, overwrite=True)