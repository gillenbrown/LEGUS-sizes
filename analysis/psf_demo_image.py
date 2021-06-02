"""
psf_demo_image.py - Make the image of the PSFs that will go in the paper

Takes the following parameters:
- Path to save the plot
- Oversampling factor used for the PSF
- paths to all the PSFs
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
import betterplotlib as bpl

bpl.set_style()

# ======================================================================================
#
# get the psfs the user has passed in
#
# ======================================================================================
plot_path = Path(sys.argv[1])
oversampling_factor = int(sys.argv[2])
psfs = dict()
for psf_path_str in sys.argv[3:]:
    # store the psf in the dictionary above based on its name
    psf_path = Path(psf_path_str)
    psf = fits.open(psf_path)["PRIMARY"].data
    galaxy_name = psf_path.parent.parent.name
    psfs[galaxy_name] = psf

# pick the galaxy to highlight
galaxy_to_show = "ngc1313-e"

# ======================================================================================
#
# Functions to set up the plot
#
# ======================================================================================
# This will have the radial profiles of all PSFs on the left, and the image of one on
# the right
cmap = bpl.cm.lapaz
cmap.set_bad(cmap(0))  # for negative values in log plot
vmax = 0.035
vmin = 1e-6


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def bin(bin_size, xs, ys):
    # then sort them by xs first
    idxs_sort = np.argsort(xs)
    xs = np.array(xs)[idxs_sort]
    ys = np.array(ys)[idxs_sort]

    # then go through and put them in bins
    binned_xs = []
    binned_ys = []
    this_bin_ys = []
    max_bin = bin_size  # start at zero
    for idx in range(len(xs)):
        x = xs[idx]
        y = ys[idx]

        # see if we need a new max bin
        if x > max_bin:
            # store our saved data
            if len(this_bin_ys) > 0:
                binned_ys.append(np.mean(this_bin_ys))
                binned_xs.append(max_bin - 0.5 * bin_size)
            # reset the bin
            this_bin_ys = []
            max_bin = np.ceil(x / bin_size) * bin_size

        assert x <= max_bin
        this_bin_ys.append(y)

    return np.array(binned_xs), np.array(binned_ys)


def radial_profile_psf(psf):
    # the center is the central pixel of the image
    x_cen = int((psf.shape[1] - 1.0) / 2.0)
    y_cen = int((psf.shape[0] - 1.0) / 2.0)
    # then go through all the pixel values to determine the distance from the center
    radii = []
    values = []
    for x in range(psf.shape[1]):
        for y in range(psf.shape[1]):
            # need to include the oversampling factor in the distance
            radii.append(distance(x, y, x_cen, y_cen) / oversampling_factor)
            values.append(psf[y][x])

    assert np.isclose(np.sum(values), 1.0)
    # then bin then
    radii, values = bin(0.1, radii, values)

    return radii, values


# ======================================================================================
#
# Then the plot itself
#
# ======================================================================================
# vertical version
# fig = plt.figure(figsize=[6, 10.5])
# gs = gridspec.GridSpec(
#     ncols=20, nrows=20, left=0, right=1, bottom=0.07, top=1, hspace=0, wspace=0
# )
# ax0 = fig.add_subplot(gs[:10, :], projection="bpl")
# ax1 = fig.add_subplot(gs[11:, 4:19], projection="bpl")
# horizontal version
# fig = plt.figure(figsize=[9, 4])
# gs = gridspec.GridSpec(
#     ncols=40, nrows=20, left=0.01, right=0.97, bottom=0.03, top=0.99, hspace=0, wspace=0
# )
# ax0 = fig.add_subplot(gs[:, :22], projection="bpl")
# ax1 = fig.add_subplot(gs[:17, 27:], projection="bpl")
fig = plt.figure(figsize=[9, 4])
gs = gridspec.GridSpec(
    ncols=40,
    nrows=1,
    left=0.12,
    right=0.97,
    bottom=0.2,
    top=0.99,
    hspace=0,
    wspace=0,
)
ax_r = fig.add_subplot(gs[:, :19], projection="bpl")
ax_v = fig.add_subplot(gs[:, 18:], projection="bpl")

# show one of the PSFs in the right panel
norm = colors.LogNorm(vmin=vmin, vmax=vmax)
im_data = ax_v.imshow(psfs[galaxy_to_show], norm=norm, cmap=cmap, origin="lower")
fig.colorbar(im_data, ax=ax_v, pad=0)
ax_v.remove_labels("both")

# then in the right panel show the main galaxy, plus the range of other PSFs.
# plot the interesting galaxy, and use it to initialize the range
radii, values = radial_profile_psf(psfs[galaxy_to_show])
ax_r.plot(radii, values, color=bpl.almost_black, lw=3, zorder=5)
max_psfs = values
min_psfs = values
# then go through each galaxy and determine expand the range as needed
for galaxy in psfs:
    radii, values = radial_profile_psf(psfs[galaxy])
    max_psfs = np.maximum(max_psfs, values)
    min_psfs = np.minimum(min_psfs, values)
ax_r.fill_between(x=radii, y1=min_psfs, y2=max_psfs, color="0.8", zorder=1)

ax_r.add_labels("Radius (pixels)", "Normalized Pixel Value")
ax_r.set_limits(0, 10, 1e-6, 0.035)
ax_r.set_yscale("log")
ax_r.xaxis.set_ticks([0, 2, 4, 6, 8, 10])

fig.savefig(plot_path)
