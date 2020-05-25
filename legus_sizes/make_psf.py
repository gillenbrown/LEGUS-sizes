"""
make_psf.py

Reads in the stars from `select_psf_stars.py` and uses them to make an empirical PSF.
"""
# https://photutils.readthedocs.io/en/stable/epsf.html
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy import table
from astropy import stats
from astropy.nddata import NDData
import photutils
import betterplotlib as bpl
from matplotlib import colors

import utils

bpl.set_style()

# ======================================================================================
#
# Get the parameters the user passed in
#
# ======================================================================================
# start by getting the output name, which we can use to get the home directory
psf_name = Path(sys.argv[1]).absolute()
size_home_dir = psf_name.parent
home_dir = size_home_dir.parent
# We'll need to get the star list too
star_list_path = Path(sys.argv[2]).absolute()
oversampling_factor = int(sys.argv[3])

# ======================================================================================
#
# Load the data - image and star catalog
#
# ======================================================================================
image_data, instrument = utils.get_f555w_drc_image(home_dir)
# the extract_stars thing below requires the input as a NDData object
nddata = NDData(data=image_data)

# get the noise_level, which will be used later
_, _, noise = stats.sigma_clipped_stats(image_data, sigma=2.0)

# load the input catalog
star_table = table.Table.read(star_list_path, format="ascii.ecsv")
# rename columns, as required by extract_stars
star_table.rename_columns(["xcentroid", "ycentroid"], ["x", "y"])

# ======================================================================================
#
# make the cutouts
#
# ======================================================================================
star_cutouts = photutils.psf.extract_stars(nddata, star_table, size=31)

# plot these for the user to see
ncols = 5
nrows = int(np.ceil(len(star_cutouts) / 5))

fig, axs = bpl.subplots(
    nrows=nrows, ncols=ncols, figsize=[4 * ncols, 3 * nrows], tight_layout=True
)
axs = axs.flatten()

for ax, cutout in zip(axs, star_cutouts):
    vmax = np.max(cutout)
    vmin = -5 * noise
    linthresh = max(0.01 * vmax, 5 * noise)
    norm = colors.SymLogNorm(vmin=vmin, vmax=vmax * 2, linthresh=linthresh)
    im = ax.imshow(cutout, norm=norm, cmap=bpl.cm.lapaz)
    ax.remove_labels("both")
    ax.remove_spines(["all"])
    fig.colorbar(im, ax=ax)

fig.savefig(size_home_dir / "psf_stars.png", dpi=100, bbox_inches="tight")

# ======================================================================================
#
# then combine to make the PSF
#
# ======================================================================================
# we don't need many iterations, since our star finder already found the centroids.
# I experimented with these values to produce the best-looking PSFs, although this was
# done with a sloppier selection of stars, this remains to be fine-tuned later
psf_builder = photutils.EPSFBuilder(
    oversampling=oversampling_factor, maxiters=3, progress_bar=True
)
psf, fitted_stars = psf_builder(star_cutouts)

psf_data = psf.data

# ======================================================================================
#
# Plot it
#
# ======================================================================================
fig, ax = bpl.subplots()
vmax = np.max(psf_data)
vmin = np.min(psf_data)
norm = colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=abs(vmin))
im = ax.imshow(psf_data, norm=norm, cmap=bpl.cm.lapaz)
ax.remove_labels("both")
ax.remove_spines(["all"])
fig.colorbar(im, ax=ax)

fig.savefig(size_home_dir / "psf.png", bbox_inches="tight")

# ======================================================================================
#
# then save it as a fits file
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(psf.data)
new_hdu.writeto(psf_name, overwrite=True)
