"""
make_psf.py - Uses the previously-created list of stars to generate a new psf.

Takes the following command line arguments:
- Name to save the PSF as.
- Oversampling factor
- Pixel size for the PSF snapshot
- The source of the coordinate lists. Must either be "my" or "legus"
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
star_source = sys.argv[4]

# check that the source is correct
if not star_source in ["my", "legus"]:
    raise ValueError("Bad final parameter to make_psf.py. Must be 'my' or 'legus'")

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

# load the input star list. This depends on what source we have for these stars
if star_source == "my":
    star_table = table.Table.read(size_home_dir / "psf_stars.txt", format="ascii.ecsv")
    # rename columns, as required by extract_stars
    star_table.rename_columns(["xcentroid", "ycentroid"], ["x", "y"])
else:
    # For the LEGUS list I'll do this myself because of the formatting. We do have
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

top_inches = 1.5
inches_per_row = 3
inches_per_col = 3.4
fig, axs = bpl.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=[inches_per_col * ncols, top_inches + inches_per_row * nrows],
    tight_layout=False,
    gridspec_kw={
        "top": (inches_per_row * nrows) / (top_inches + inches_per_row * nrows),
        "wspace": 0.18,
        "hspace": 0.35,
        "left": 0.01,
        "right": 0.98,
        "bottom": 0.01,
    },
)
axs = axs.flatten()
for ax in axs:
    ax.set_axis_off()

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

# reformat the name
if star_source == "my":
    plot_title = f"{str(home_dir.name).upper()} - Me"
else:
    plot_title = f"{str(home_dir.name).upper()} - LEGUS"

fig.suptitle(plot_title, fontsize=40)

figname = (
    f"psf_stars_{star_source}_"
    + f"{psf_width}_pixels_"
    + f"{oversampling_factor}x_oversampled.png"
)

fig.savefig(
    size_home_dir / "plots" / figname, dpi=100,
)

# ======================================================================================
#
# then combine to make the PSF
#
# ======================================================================================
psf_builder = photutils.EPSFBuilder(
    oversampling=oversampling_factor, maxiters=3, progress_bar=True
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
fig, axs = bpl.subplots(
    ncols=2,
    figsize=[12, 5],
    tight_layout=False,
    gridspec_kw={"top": 0.9, "left": 0.05, "right": 0.95, "wspace": 0.1},
)

vmax = np.max(psf_data)
norm_log = colors.SymLogNorm(vmin=0, vmax=vmax, linthresh=0.02 * vmax, base=10)
norm_lin = colors.Normalize(vmin=0, vmax=vmax)

im_lin = axs[0].imshow(psf_data, norm=norm_lin, cmap=bpl.cm.lapaz)
im_log = axs[1].imshow(psf_data, norm=norm_log, cmap=bpl.cm.lapaz)

fig.colorbar(im_lin, ax=axs[0])
fig.colorbar(im_log, ax=axs[1])

for ax in axs:
    ax.remove_labels("both")
    ax.remove_spines(["all"])
fig.suptitle(plot_title, fontsize=24)

figname = (
    f"psf_{star_source}_"
    + f"{psf_width}_pixels_"
    + f"{oversampling_factor}x_oversampled.png"
)
fig.savefig(size_home_dir / "plots" / figname, bbox_inches="tight")

# ======================================================================================
#
# then save it as a fits file
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(psf.data)
new_hdu.writeto(psf_name, overwrite=True)
