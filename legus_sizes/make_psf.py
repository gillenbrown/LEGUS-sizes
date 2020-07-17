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
# background subtract them. Here I use the pixels farther than 8 pixels from the center.
# this value was determined by looking at the profiles.
def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


for star in star_cutouts:
    x_cen = star.cutout_center[0]  # yes, this indexing is correct, look at the docs or
    y_cen = star.cutout_center[1]  # at the bottom of this and psf_compare.py to see use
    border_pixels = [
        star.data[y][x]
        for x in range(star.data.shape[1])
        for y in range(star.data.shape[0])
        if distance(x_cen, y_cen, x, y) > 8
    ]
    # quick estimate of how many pixels we expect to have here. 0.9 is fudge factor
    assert len(border_pixels) > 0.9 * (psf_width ** 2 - np.pi * 8 ** 2)
    star._data = star.data - np.median(border_pixels)

# ======================================================================================
#
# then combine to make the PSF
#
# ======================================================================================
psf_builder = photutils.EPSFBuilder(
    oversampling=oversampling_factor,
    maxiters=20,
    smoothing_kernel="quadratic",  # more stable than quartic
    progress_bar=False,
)
psf, fitted_stars = psf_builder(star_cutouts)

psf_data = psf.data
# the convolution requires the psf to be normalized, and without any negative values
psf_data = np.maximum(psf_data, 0)
psf_data /= np.sum(psf_data)

# ======================================================================================
#
# Plot the cutouts
#
# ======================================================================================
cmap = bpl.cm.lapaz
cmap.set_bad(cmap(0))  # for negative values in log plot

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

for ax, cutout, row, fitted_star in zip(axs, star_cutouts, star_table, fitted_stars):
    vmax = np.max(cutout)
    vmin = -5 * noise
    linthresh = max(0.01 * vmax, 5 * noise)
    norm = colors.SymLogNorm(vmin=vmin, vmax=vmax * 2, linthresh=linthresh, base=10)
    im = ax.imshow(cutout, norm=norm, cmap=cmap, origin="lower")
    # add a marker at the location identified as the center
    ax.scatter(
        [fitted_star.cutout_center[0]],
        fitted_star.cutout_center[1],
        c=bpl.almost_black,
        marker="x",
    )
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
    f"psf_stars_{star_source}_stars_"
    + f"{psf_width}_pixels_"
    + f"{oversampling_factor}x_oversampled.png"
)

fig.savefig(
    size_home_dir / "plots" / figname, dpi=100,
)

# ======================================================================================
#
# then save it as a fits file
#
# ======================================================================================
new_hdu = fits.PrimaryHDU(psf_data)
new_hdu.writeto(psf_name, overwrite=True)

# also save the fitted stars
x_cens = [star.cutout_center[0] + star.origin[0] for star in fitted_stars.all_stars]
y_cens = [star.cutout_center[1] + star.origin[1] for star in fitted_stars.all_stars]
fitted_star_table = table.Table([x_cens, y_cens], names=("x_center", "y_center"))
savename = (
    f"psf_star_centers_{star_source}_stars_"
    + f"{psf_width}_pixels_"
    + f"{oversampling_factor}x_oversampled.txt"
)
fitted_star_table.write(size_home_dir / savename, format="ascii.ecsv")
