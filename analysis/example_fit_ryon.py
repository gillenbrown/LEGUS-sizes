"""
example_fit.py - Plot an example showing a fitted cluster
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table, nddata
from astropy.io import fits
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import gridspec, colors
import cmocean
import betterplotlib as bpl

bpl.set_style()

# need to add the correct path to import utils
legus_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(legus_home_dir / "pipeline"))
import utils
import fit_utils

# get the location to save this plot
plot_name = Path(sys.argv[1]).resolve()
oversampling_factor = int(sys.argv[2])
psf_size = int(sys.argv[3])
snapshot_size = int(sys.argv[4])
snapshot_size_oversampled = snapshot_size * oversampling_factor

# ======================================================================================
#
# Load the data we need
#
# ======================================================================================
# Use NGC6503 ID 0059
galaxy = "ngc1313-w"  # "ngc5194-ngc5195-mosaic"
cluster_id = 1483

data_dir = legus_home_dir / "data" / galaxy
image_data, _, _ = utils.get_drc_image(data_dir)

error_data = fits.open(data_dir / "size" / "sigma_electrons.fits")["PRIMARY"].data
mask = fits.open(data_dir / "size" / "mask_image.fits")["PRIMARY"].data

psf_name = f"psf_my_stars_{psf_size}_pixels_{oversampling_factor}x_oversampled.fits"
psf = fits.open(data_dir / "size" / psf_name)["PRIMARY"].data
psf_cen = int((psf.shape[1] - 1.0) / 2.0)


cat_name = (
    f"final_catalog_final_{snapshot_size}_pixels_psf_"
    f"my_stars_{psf_size}_pixels_{oversampling_factor}x_oversampled.txt"
)
cat_path = data_dir / "size" / cat_name
cat = table.Table.read(str(cat_path), format="ascii.ecsv")
# then find the correct row
for row in cat:
    if row["ID"] == cluster_id:
        break

# Then get the snapshot of this cluster
x_cen = int(np.ceil(row["x_fitted_best"]))
y_cen = int(np.ceil(row["y_fitted_best"]))

# Get the snapshot, based on the size desired
x_min = x_cen - 15
x_max = x_cen + 15
y_min = y_cen - 15
y_max = y_cen + 15

data_snapshot = image_data[y_min:y_max, x_min:x_max]
error_snapshot = error_data[y_min:y_max, x_min:x_max]
mask_snapshot = mask[y_min:y_max, x_min:x_max]
# Use the same mask region as was used in the actual fitting procedure
mask_snapshot = fit_utils.handle_mask(mask_snapshot, row["ID"])

# then have new centers for the fit within this snapshot. See the code in fit.py to
# correct for the oversampling factor
x_cen_snap = row["x_fitted_best"] - x_min
y_cen_snap = row["y_fitted_best"] - y_min
x_cen_snap_oversampled = (x_cen_snap + 0.25) * 2
y_cen_snap_oversampled = (y_cen_snap + 0.25) * 2

# get the Ryon parameters for this cluster
if galaxy.startswith("ngc1313"):
    ryon_cat = table.Table.read("ryon_results_ngc1313.txt", format="ascii.cds")
elif galaxy.starswith("ngc628"):
    ryon_cat = table.Table.read("ryon_results_ngc628.txt", format="ascii.cds")
else:
    raise ValueError("Must use either NGC1313 or NGC628")
# then get the params for this cluster
ryon_like_id = str(cluster_id) + galaxy[-2:]
for row_ryon in ryon_cat:
    if row_ryon["ID"] == ryon_like_id:
        # effective radius is in parsecs. Need to get this into pixels. To do this I'll
        # use my utilities strangely, and see what 1 pixel would be in pc, then divide
        arcsec_per_pix = utils.pixels_to_arcsec(1, data_dir)
        pc_per_pix = utils.arcsec_to_pc_with_errors(
            data_dir, arcsec_per_pix, 0, 0, ryon=True
        )[0]

        r_eff_ryon_pc = 10 ** row_ryon["logReff-gal"]
        r_eff_ryon_pix = r_eff_ryon_pc / pc_per_pix

        eta_ryon = row_ryon["Eta"]
        # reverse engineer this to get scale radius. Equation 11 in my paper
        a_ryon = r_eff_ryon_pix / np.sqrt(0.5 ** (1 / (1 - eta_ryon)) - 1)

# ======================================================================================
#
# Creating the EFF profile
#
# ======================================================================================
models_mine = fit_utils.create_model_image(
    row["log_luminosity_best"],
    x_cen_snap_oversampled,
    y_cen_snap_oversampled,
    row["scale_radius_pixels_best"],
    row["axis_ratio_best"],
    row["position_angle_best"],
    row["power_law_slope_best"],
    row["local_background_best"],
    psf,
    snapshot_size_oversampled,
    oversampling_factor,
)
model_image_mine, model_psf_image_mine, model_psf_bin_image_mine = models_mine

sigma_snapshot_mine = (data_snapshot - model_psf_bin_image_mine) / error_snapshot
sigma_snapshot_mine *= mask_snapshot

# and do the same for the Ryon model. I'll reuse some of the simple parameters
models_ryon = fit_utils.create_model_image(
    row["log_luminosity_best"],
    x_cen_snap_oversampled,
    y_cen_snap_oversampled,
    a_ryon,
    row["axis_ratio_best"],
    row["position_angle_best"],
    eta_ryon,
    row["local_background_best"],
    psf,
    snapshot_size_oversampled,
    oversampling_factor,
)
model_image_ryon, model_psf_image_ryon, model_psf_bin_image_ryon = models_ryon

sigma_snapshot_ryon = (data_snapshot - model_psf_bin_image_ryon) / error_snapshot
sigma_snapshot_ryon *= mask_snapshot
# ======================================================================================
#
# Convenience functions for the plot
#
# ======================================================================================
# These don't really need to be functions, but it makes things cleaner in the plot below


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def radial_profile(snapshot, oversampling_factor, x_c, y_c):
    radii, ys = [], []
    for x in range(snapshot.shape[1]):
        for y in range(snapshot.shape[0]):
            radii.append(distance(x, y, x_c, y_c) / oversampling_factor)
            ys.append(snapshot[y, x])
    idx_sort = np.argsort(radii)
    return np.array(radii)[idx_sort], np.array(ys)[idx_sort]


def binned_radial_profile(snapshot, oversampling_factor, x_c, y_c, bin_size):
    radii, ys = radial_profile(snapshot, oversampling_factor, x_c, y_c)
    # then bin this data to make the binned plot
    if radii[0] == 0.0:
        binned_radii = [0]
        binned_ys = [ys[0]]
        radii = radii[1:]
        ys = ys[1:]
    else:
        binned_radii, binned_ys = [], []

    for r_min in np.arange(0, int(np.ceil(max(radii))), bin_size):
        r_max = r_min + bin_size
        idx_above = np.where(r_min < radii)
        idx_below = np.where(r_max > radii)
        idx_good = np.intersect1d(idx_above, idx_below)

        if len(idx_good) > 0:
            binned_radii.append(r_min + 0.5 * bin_size)
            binned_ys.append(np.mean(ys[idx_good]))
    return binned_radii, binned_ys


# ======================================================================================
#
# Make the plot
#
# ======================================================================================
# params to use elsewhere
data_vmin, data_vmax = 5e2, 2e4
sigma_max = 10
plot_min, plot_max = 500, 1e5
r_eff_ymax = 3e4  # how high the r_eff line goes
r_eff_label_y = 1e3
# vmax = max(np.max(model_image), np.max(model_psf_image), np.max(data_snapshot))
data_norm = colors.LogNorm(vmin=data_vmin, vmax=data_vmax)
sigma_norm = colors.Normalize(vmin=-sigma_max, vmax=sigma_max)
data_cmap = bpl.cm.davos
data_cmap.set_bad(data_cmap(0))
sigma_cmap = cmocean.cm.tarn  # "bwr_r" also works

# This will have the data, model, and residual above the plot
fig = plt.figure(figsize=[20, 7])
gs = gridspec.GridSpec(
    nrows=2,
    ncols=5,
    width_ratios=[2, 1, 1, 1, 1],
    wspace=0.1,
    hspace=0.2,
    left=0.08,
    right=0.98,
    bottom=0.1,
    top=0.9,
)

ax_big = fig.add_subplot(gs[:, 0], projection="bpl")  # radial profile
ax_da = fig.add_subplot(gs[0, 1], projection="bpl")  # data
ax_rm = fig.add_subplot(gs[0, 2], projection="bpl")  # raw model, me
ax_fm = fig.add_subplot(gs[0, 3], projection="bpl")  # full model (f for fit), me
ax_sm = fig.add_subplot(gs[0, 4], projection="bpl")  # sigma difference, me
ax_rr = fig.add_subplot(gs[1, 2], projection="bpl")  # raw model, Ryon
ax_fr = fig.add_subplot(gs[1, 3], projection="bpl")  # full model (f for fit), Ryon
ax_sr = fig.add_subplot(gs[1, 4], projection="bpl")  # sigma difference, Ryon

data_common = {"origin": "lower", "cmap": data_cmap, "norm": data_norm}
sigma_common = {"origin": "lower", "cmap": sigma_cmap, "norm": sigma_norm}
da_im = ax_da.imshow(data_snapshot, **data_common)
rm_im = ax_rm.imshow(model_image_mine, **data_common)
rr_im = ax_rr.imshow(model_image_ryon, **data_common)
fm_im = ax_fm.imshow(model_psf_bin_image_mine, **data_common)
fr_im = ax_fr.imshow(model_psf_bin_image_ryon, **data_common)
sm_im = ax_sm.imshow(sigma_snapshot_mine, **sigma_common)
sr_im = ax_sr.imshow(sigma_snapshot_ryon, **sigma_common)

fig.colorbar(da_im, ax=ax_da, pad=0)
fig.colorbar(rm_im, ax=ax_rm, pad=0)
fig.colorbar(rr_im, ax=ax_rr, pad=0)
fig.colorbar(fm_im, ax=ax_fm, pad=0)
fig.colorbar(fr_im, ax=ax_fr, pad=0)
fig.colorbar(sm_im, ax=ax_sm, pad=0)
fig.colorbar(sr_im, ax=ax_sr, pad=0)

title_fontsize = 16
ax_da.set_title("Data", fontsize=title_fontsize)
ax_rm.set_title("Raw Cluster Model", fontsize=title_fontsize)
ax_fm.set_title("Model Convolved\nWith PSF", fontsize=title_fontsize)
ax_sm.set_title("(Data - Model)/Uncertainty", fontsize=title_fontsize)

for ax in [ax_da, ax_rm, ax_fm, ax_sm, ax_rr, ax_fr, ax_sr]:
    ax.remove_labels("both")
    ax.remove_spines(["all"])

# Then the radial profiles
ax_big.plot(
    *binned_radial_profile(
        model_image_mine, 2, x_cen_snap_oversampled, y_cen_snap_oversampled, 0.1
    ),
    label="Raw Cluster Model - This Work",
    c=bpl.color_cycle[3],
    lw=4,
)
ax_big.plot(
    *binned_radial_profile(
        model_image_ryon, 2, x_cen_snap_oversampled, y_cen_snap_oversampled, 0.1
    ),
    label="Raw Cluster Model - Ryon+17",
    c=bpl.color_cycle[3],
    lw=4,
)
ax_big.plot(
    *binned_radial_profile(model_psf_bin_image_mine, 1, x_cen_snap, y_cen_snap, 0.25),
    label="Model Convolved with PSF - This Work",
    c=bpl.color_cycle[0],
    lw=4,
)
ax_big.plot(
    *binned_radial_profile(model_psf_bin_image_mine, 1, x_cen_snap, y_cen_snap, 0.25),
    label="Model Convolved with PSF - Ryon+17",
    c=bpl.color_cycle[0],
    lw=4,
)
ax_big.scatter(
    *radial_profile(data_snapshot, 1, x_cen_snap, y_cen_snap),
    label="Data",
    c=bpl.color_cycle[2],
)

# Use the final model to normalize the psf
psf *= np.max(model_psf_bin_image_mine) / np.max(psf)
ax_big.plot(
    *binned_radial_profile(psf, 2, psf_cen, psf_cen, 0.1),
    label="PSF",
    c=bpl.color_cycle[1],
    lw=4,
)


ax_big.axhline(row["local_background_best"], ls=":", label="Local Background")
# ax_big.add_text(
#     x=5.5,
#     y=row["local_background_best"] * 1.1,
#     text="Local Background",
#     ha="left",
#     va="bottom",
#     fontsize=18,
# )

r_eff = row["r_eff_pixels_rmax_15pix_best"]
ax_big.plot([r_eff, r_eff], [0, r_eff_ymax], c=bpl.almost_black, ls="--")
ax_big.add_text(
    x=r_eff - 0.05,
    y=r_eff_label_y,
    text="$R_{eff}$",
    ha="right",
    va="bottom",
    rotation=90,
    fontsize=25,
)

# y_at_rmax = profile_at_radius(15)
# ax_big.plot([15, 15], [0, y_at_rmax], c=bpl.almost_black, ls="--")
# ax_big.add_text(
#     x=15 - 0.05,
#     y=y_at_rmax * 0.8,
#     text="$R_{max}$",
#     ha="right",
#     va="top",
#     rotation=90,
#     fontsize=25,
# )

ax_big.legend()
ax_big.set_yscale("log")
ax_big.add_labels("Radius [pixels]", "Pixel Value [e$^-$]")
max_pix = 7
ax_big.set_limits(0, max_pix, plot_min, plot_max)

# then add a second scale on top translating into parsecs
arcsec = utils.pixels_to_arcsec(max_pix, data_dir)
max_pc, _, _ = utils.arcsec_to_pc_with_errors(data_dir, arcsec, 0, 0)
ax_big.twin_axis_simple("x", lower_lim=0, upper_lim=max_pc, label="Radius [pc]")

fig.savefig(plot_name)
