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
oversampling_factor = int(sys.argv[2])
psf_width = int(sys.argv[3])

# ======================================================================================
#
# Load the data - image and star catalog
#
# ======================================================================================
image_data, _, _ = utils.get_drc_image(home_dir)
# the extract_stars thing below requires the input as a NDData object
nddata = NDData(data=image_data)

# get the noise_level, which will be used later
_, _, noise = stats.sigma_clipped_stats(image_data, sigma=2.0)

# load the input star list. I'll do this myself because of the formatting. We do have
# to be careful with one galaxy which has a long filename
galaxy_name = home_dir.name
if galaxy_name == "ngc5194-ngc5195-mosaic":
    name = "isolated_stars__f435w_f555w_f814w_ngc5194-ngc5195-mosaic.coo"
else:
    name = f"isolated_stars_{galaxy_name}.coo"
preliminary_catalog_path = home_dir / name

star_table = table.Table(names=("x", "y"))
with open(preliminary_catalog_path, "r") as in_file:
    for row in in_file:
        row = row.strip()
        if (not row.startswith("#")) and row != "":
            star_table.add_row((row.split()[0], row.split()[1]))


# handle one vs zero indexing
star_table["x"] -= 1
star_table["y"] -= 1

# ======================================================================================
#
# make the cutouts
#
# ======================================================================================
star_cutouts = photutils.psf.extract_stars(nddata, star_table, size=psf_width)

# plot these for the user to see
ncols = 5
nrows = int(np.ceil(len(star_cutouts) / 5))

fig, axs = bpl.subplots(
    nrows=nrows, ncols=ncols, figsize=[3.5 * ncols, 3 * nrows], tight_layout=True
)
axs = axs.flatten()

for ax, cutout, row in zip(axs, star_cutouts, star_table):
    vmax = np.max(cutout)
    vmin = -5 * noise
    linthresh = max(0.01 * vmax, 5 * noise)
    norm = colors.SymLogNorm(vmin=vmin, vmax=vmax * 2, linthresh=linthresh, base=10)
    im = ax.imshow(cutout, norm=norm, cmap=bpl.cm.lapaz)
    ax.remove_labels("both")
    ax.remove_spines(["all"])
    ax.set_title(f"x={row['x']:.0f}\ny={row['y']:.0f}")
    fig.colorbar(im, ax=ax)

fig.suptitle(str(home_dir.name).upper(), fontsize=20)

fig.savefig(size_home_dir / "plots" / "psf_stars.png", dpi=100, bbox_inches="tight")

# ======================================================================================
#
# then combine to make the PSF
#
# ======================================================================================
psf_builder = photutils.EPSFBuilder(
    oversampling=oversampling_factor, maxiters=10, progress_bar=True
)
psf, fitted_stars = psf_builder(star_cutouts)

psf_data = psf.data
# the convolution requires the psf to be normalized, and without any negative values
psf_data = np.maximum(psf_data, 0)
psf_data /= np.sum(psf_data)

# ======================================================================================
#
# Plot it
#
# ======================================================================================
fig, ax = bpl.subplots()
vmax = np.max(psf_data)
vmin = np.min(psf_data)
norm = colors.SymLogNorm(vmin=0, vmax=0.1, linthresh=0.002, base=10)
im = ax.imshow(psf_data, norm=norm, cmap=bpl.cm.lapaz)
ax.remove_labels("both")
ax.remove_spines(["all"])
ax.set_title(str(home_dir.name).upper())
fig.colorbar(im, ax=ax)

fig.savefig(size_home_dir / "plots" / "psf.png", bbox_inches="tight")

# ======================================================================================
#
# then save it as a fits file
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(psf.data)
new_hdu.writeto(psf_name, overwrite=True)
