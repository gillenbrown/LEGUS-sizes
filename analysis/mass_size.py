"""
mass_size.py - plot the mass-size relation for LEGUS clusters
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits
import betterplotlib as bpl

# need to add the correct path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent / "pipeline"))
import utils

bpl.set_style()

# ======================================================================================
#
# load the parameters the user passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1])
oversampling_factor = int(sys.argv[2])
psf_width = int(sys.argv[3])
psf_source = sys.argv[4]
catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[5:]]
# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# Calculate the fractional error
big_catalog["fractional_err-"] = (
    big_catalog["r_eff_pc_rmax_15pix_e-"] / big_catalog["r_eff_pc_rmax_15pix_best"]
)
big_catalog["fractional_err+"] = (
    big_catalog["r_eff_pc_rmax_15pix_e+"] / big_catalog["r_eff_pc_rmax_15pix_best"]
)
big_catalog["fractional_err_max"] = np.maximum(
    big_catalog["fractional_err-"], big_catalog["fractional_err+"]
)

# then filter out some clusters
print(f"Total Clusters: {len(big_catalog)}")
mask = big_catalog["age_yr"] <= 200e6
print(f"Clusters with age < 200 Myr: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["scale_radius_pixels_best"] > 0.1)
print(f"Clusters with scale radius > 0.1 pixel: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["scale_radius_pixels_best"] < 15.0)
print(f"Clusters with scale radius < 15 pixels: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["axis_ratio_best"] > 0.2)
print(f"Clusters with axis ratio > 0.2: {np.sum(mask)}")

# ======================================================================================
#
# make the plot
#
# ======================================================================================
def get_r_percentiles(radii, masses, percentile, d_log_M):
    bins = np.logspace(2, 7, int(5 / d_log_M) + 1)

    bin_centers = []
    radii_percentiles = []
    for idx in range(len(bins) - 1):
        lower = bins[idx]
        upper = bins[idx + 1]

        # then find all clusters in this mass range
        mask_above = masses > lower
        mask_below = masses < upper
        mask_good = np.logical_and(mask_above, mask_below)

        good_radii = radii[mask_good]
        if len(good_radii) > 0:
            radii_percentiles.append(np.percentile(good_radii, percentile))
            # the bin centers will be the mean in log space
            bin_centers.append(10 ** np.mean([np.log10(lower), np.log10(upper)]))

    return bin_centers, radii_percentiles


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def measure_psf_reff(psf):
    # the center is the central pixel of the image
    x_cen = int((psf.shape[1] - 1.0) / 2.0)
    y_cen = int((psf.shape[0] - 1.0) / 2.0)
    total = np.sum(psf)
    half_light = total / 2.0
    # then go through all the pixel values to determine the distance from the center.
    # Then we can go through them in order to determine the half mass radius
    radii = []
    values = []
    for x in range(psf.shape[1]):
        for y in range(psf.shape[1]):
            # need to include the oversampling factor in the distance
            radii.append(distance(x, y, x_cen, y_cen) / oversampling_factor)
            values.append(psf[y][x])

    idxs_sort = np.argsort(radii)
    sorted_radii = np.array(radii)[idxs_sort]
    sorted_values = np.array(values)[idxs_sort]

    cumulative_light = 0
    for idx in range(len(sorted_radii)):
        cumulative_light += sorted_values[idx]
        if cumulative_light >= half_light:
            return sorted_radii[idx]


masses = big_catalog["mass_msun"][mask]
for unit in ["pc", "pixels"]:
    radii = big_catalog[f"r_eff_{unit}_rmax_15pix_best"][mask]
    fig, ax = bpl.subplots()

    ax.scatter(masses, radii, alpha=1.0, s=2)
    if unit == "pc":
        # plot the median and the IQR
        d_log_M = 0.25
        for percentile in [5, 10, 25, 50, 75, 90, 95]:
            mass_bins, radii_percentile = get_r_percentiles(
                radii, masses, percentile, d_log_M
            )
            ax.plot(
                mass_bins,
                radii_percentile,
                c=bpl.almost_black,
                lw=4 * (1 - (abs(percentile - 50) / 50)) + 0.5,
                zorder=1,
            )
            ax.text(
                x=mass_bins[0],
                y=radii_percentile[0],
                ha="center",
                va="bottom",
                s=percentile,
                fontsize=16,
            )

    # then add all the PSF widths. Here we load the PSF and directly measure it's R_eff,
    # so we can have a fair comparison to the clusters
    for cat_loc in sys.argv[5:]:
        size_home_dir = Path(cat_loc).parent
        home_dir = size_home_dir.parent

        psf_name = (
            f"psf_"
            f"{psf_source}_stars_"
            f"{psf_width}_pixels_"
            f"{oversampling_factor}x_oversampled.fits"
        )

        psf = fits.open(size_home_dir / psf_name)["PRIMARY"].data
        psf_size_pixels = measure_psf_reff(psf)
        if unit == "pc":
            psf_size_arcsec = utils.pixels_to_arcsec(psf_size_pixels, home_dir)
            psf_size_pc = utils.arcsec_to_pc_with_errors(
                home_dir, psf_size_arcsec, 0, 0, False
            )[0]
        ax.plot(
            [7e5, 1e6], [psf_size_pc, psf_size_pc], lw=1, c=bpl.almost_black, zorder=3
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_limits(1e2, 1e6, 0.2, 40)
    ax.add_labels("Cluster Mass [M$_\odot$]", f"Cluster Effective Radius [{unit}]")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    if unit == "pc":
        fig.savefig(plot_name)
    else:
        new_name = plot_name.name.strip(".png")
        new_name += "pix.png"
        fig.savefig(plot_name.parent / new_name)


# ======================================================================================
#
# similar plot
#
# ======================================================================================
fig, ax = bpl.subplots()

# then add all the PSF widths. Here we load the PSF and directly measure it's R_eff,
# so we can have a fair comparison to the clusters
all_ratios = np.array([])
for cat_loc in sys.argv[5:]:
    cat = table.Table.read(cat_loc, format="ascii.ecsv")

    size_home_dir = Path(cat_loc).parent
    home_dir = size_home_dir.parent

    psf_name = (
        f"psf_"
        f"{psf_source}_stars_"
        f"{psf_width}_pixels_"
        f"{oversampling_factor}x_oversampled.fits"
    )

    psf = fits.open(size_home_dir / psf_name)["PRIMARY"].data
    psf_size = measure_psf_reff(psf)

    this_ratio = cat[f"r_eff_pixels_rmax_15pix_best"].data / psf_size
    all_ratios = np.concatenate([all_ratios, this_ratio])

ax.hist(
    all_ratios, alpha=1.0, lw=1, color=bpl.color_cycle[3], bins=np.logspace(-1, 1, 21),
)

ax.axvline(1.0)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_limits(0.1, 10)
ax.add_labels("Cluster Effective Radius / PSF Effective Radius", "Number of Clusters")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
fig.savefig(plot_name.parent / "r_eff_over_psf.png", bbox_inches="tight")
