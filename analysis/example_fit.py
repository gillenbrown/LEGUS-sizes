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

# get the location to save this plot
plot_name = Path(sys.argv[1]).resolve()

# ======================================================================================
#
# Load the data we need
#
# ======================================================================================
# Use NGC6503 ID 0059
galaxy = "ic559"
cluster_id = 241

data_dir = legus_home_dir / "data" / galaxy
image_data, _, _ = utils.get_drc_image(data_dir)

error_data = fits.open(data_dir / "size" / "sigma_electrons.fits")["PRIMARY"].data

psf_name = f"psf_my_stars_15_pixels_2x_oversampled.fits"
psf = fits.open(data_dir / "size" / psf_name)["PRIMARY"].data
psf_cen = int((psf.shape[1] - 1.0) / 2.0)


cat_name = "final_catalog_30_pixels_psf_my_stars_15_pixels_2x_oversampled.txt"
cat_path = data_dir / "size" / cat_name
cat = table.Table.read(str(cat_path), format="ascii.ecsv")
# then find the correct row
for row in cat:
    if row["ID"] == cluster_id:
        break

# Then get the snapshot of this cluster
x_cen = int(np.ceil(row["x_pix_single_fitted_best"]))
y_cen = int(np.ceil(row["y_pix_single_fitted_best"]))

# Get the snapshot, based on the size desired
x_min = x_cen - 15
x_max = x_cen + 15
y_min = y_cen - 15
y_max = y_cen + 15

data_snapshot = image_data[y_min:y_max, x_min:x_max]
error_snapshot = error_data[y_min:y_max, x_min:x_max]

# then have new centers for the fit within this snapshot. See the code in fit.py to
# correct for the oversampling factor
x_cen_snap = row["x_pix_single_fitted_best"] - x_min
y_cen_snap = row["y_pix_single_fitted_best"] - y_min
x_cen_snap_oversampled = (x_cen_snap + 0.25) * 2
y_cen_snap_oversampled = (y_cen_snap + 0.25) * 2

# ======================================================================================
#
# Creating the EFF profile
#
# ======================================================================================
def eff_profile_2d(x, y, log_mu_0, x_c, y_c, a, q, theta, eta):
    """
    2-dimensional EFF profile, in pixel units

    :param x: X pixel values
    :param y: Y pixel values
    :param log_mu_0: Log of the central surface brightness
    :param x_c: Center X pixel coordinate
    :param y_c: Center Y pixel coordinate
    :param a: Scale radius in the major axis direction
    :param q: Axis ratio. The small axis will have scale length qa
    :param theta: Position angle
    :param eta: Power law slope
    :return: Values of the function at the x,y coordinates passed in
    """
    x_prime = (x - x_c) * np.cos(theta) + (y - y_c) * np.sin(theta)
    y_prime = -(x - x_c) * np.sin(theta) + (y - y_c) * np.cos(theta)

    x_term = (x_prime / a) ** 2
    y_term = (y_prime / (q * a)) ** 2
    return (10 ** log_mu_0) * (1 + x_term + y_term) ** (-eta)


def convolve_with_psf(in_array):
    """ Convolve an array with the PSF """
    # Scipy FFT based convolution was tested to be the fastest. It does have edge
    # affects, but those are minimized by using our padding. We do have to modify the
    # psf to have
    return signal.fftconvolve(in_array, psf, mode="same")


def bin_data_2d(data):
    """ Bin a 2d array into square bins determined by the oversampling factor """
    # Astropy has a convenient function to do this
    bin_factors = [2, 2]
    return nddata.block_reduce(data, bin_factors, np.mean)


def create_model_image(log_mu_0, x_c, y_c, a, q, theta, eta, background):
    """ Create a model image using the EFF parameters. """
    # a is passed in in regular coordinates, shift it to the oversampled ones
    a *= 2

    # first generate the x and y pixel coordinates of the model image. We will have
    # an array that's the same size as the cluster snapshot in oversampled pixels,
    # plus padding to account for zero-padded boundaries in the FFT convolution
    padding = 5 * 2  # 5 regular pixels on each side
    box_length = 60 + 2 * padding

    # correct the center to be at the center of this new array
    x_c_internal = x_c + padding
    y_c_internal = y_c + padding

    x_values = np.zeros([box_length, box_length])
    y_values = np.zeros([box_length, box_length])

    for x in range(box_length):
        x_values[:, x] = x
    for y in range(box_length):
        y_values[y, :] = y

    model_image = eff_profile_2d(
        x_values, y_values, log_mu_0, x_c_internal, y_c_internal, a, q, theta, eta
    )
    # convolve without the background first, to do an ever better job avoiding edge
    # effects, as the model should be zero near the boundaries anyway, matching the zero
    # padding scipy does.
    model_psf_image = convolve_with_psf(model_image)
    # model_image += background
    model_psf_image += background

    # crop out the padding before binning the data
    model_image = model_image[padding:-padding, padding:-padding]
    model_psf_image = model_psf_image[padding:-padding, padding:-padding]
    model_psf_bin_image = bin_data_2d(model_psf_image)

    # return all of these, since we'll want to use them when plotting
    return model_image, model_psf_image, model_psf_bin_image


models = create_model_image(
    np.log10(row["central_surface_brightness_best"]),
    x_cen_snap_oversampled,
    y_cen_snap_oversampled,
    row["scale_radius_pixels_best"],
    row["axis_ratio_best"],
    row["position_angle_best"],
    row["power_law_slope_best"],
    row["local_background_best"],
)
model_image, model_psf_image, model_psf_bin_image = models

sigma_snapshot = (data_snapshot - model_psf_bin_image) / error_snapshot
# ======================================================================================
#
# Convenience functions for the plot
#
# ======================================================================================
# These don't really need to be functions, but it makes things cleaner in the plot below
vmax = max(np.max(model_image), np.max(model_psf_image), np.max(data_snapshot))
data_norm = colors.LogNorm(vmin=1e-2 * vmax, vmax=vmax)
sigma_norm = colors.Normalize(vmin=-10, vmax=10)
data_cmap = bpl.cm.davos
data_cmap.set_bad(data_cmap(0))
sigma_cmap = cmocean.cm.tarn  # "bwr_r" also works


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
        binned_radii, binned_ys, = [], []

    for r_min in np.arange(0, int(np.ceil(max(radii))), bin_size):
        r_max = r_min + bin_size
        idx_above = np.where(r_min < radii)
        idx_below = np.where(r_max > radii)
        idx_good = np.intersect1d(idx_above, idx_below)

        if len(idx_good) > 0:
            binned_radii.append(r_min + 0.5 * bin_size)
            binned_ys.append(np.mean(ys[idx_good]))
    return binned_radii, binned_ys


def profile_at_radius(x):
    return eff_profile_2d(
        x=x,
        y=0,
        x_c=0,
        y_c=0,
        log_mu_0=np.log10(row["central_surface_brightness_best"]),
        a=row["scale_radius_pixels_best"],
        q=row["axis_ratio_best"],
        theta=row["position_angle_best"],
        eta=row["power_law_slope_best"],
    )


# ======================================================================================
#
# Make the plot
#
# ======================================================================================
# This will have the data, model, and residual above the plot
fig = plt.figure(figsize=[15, 12])
gs = gridspec.GridSpec(
    nrows=2,
    ncols=4,
    height_ratios=[1, 3],
    wspace=0,
    hspace=0.2,
    left=0.1,
    right=0.98,
    bottom=0.06,
    top=0.94,
)
ax_r = fig.add_subplot(gs[0, 0], projection="bpl")  # raw model
ax_f = fig.add_subplot(gs[0, 1], projection="bpl")  # full model (f for fit)
ax_d = fig.add_subplot(gs[0, 2], projection="bpl")  # data
ax_s = fig.add_subplot(gs[0, 3], projection="bpl")  # sigma difference
ax_big = fig.add_subplot(gs[1, :], projection="bpl")  # radial profile

r_im = ax_r.imshow(model_image, origin="lower", cmap=data_cmap, norm=data_norm)
f_im = ax_f.imshow(model_psf_bin_image, origin="lower", cmap=data_cmap, norm=data_norm)
d_im = ax_d.imshow(data_snapshot, origin="lower", cmap=data_cmap, norm=data_norm)
s_im = ax_s.imshow(sigma_snapshot, origin="lower", cmap=sigma_cmap, norm=sigma_norm)

fig.colorbar(r_im, ax=ax_r, pad=0)
fig.colorbar(f_im, ax=ax_f, pad=0)
fig.colorbar(d_im, ax=ax_d, pad=0)
fig.colorbar(s_im, ax=ax_s, pad=0)

ax_r.set_title("Raw Cluster Model")
ax_f.set_title("Final Model")
ax_d.set_title("Data")
ax_s.set_title("(Data - Model)/Uncertainty")

for ax in [ax_r, ax_f, ax_d, ax_s]:
    ax.remove_labels("both")
    ax.remove_spines(["all"])

# Then the radial profiles
ax_big.plot(
    *radial_profile(model_image, 2, x_cen_snap_oversampled, y_cen_snap_oversampled),
    label="Raw Cluster Model",
    c=bpl.color_cycle[3],
    lw=4,
)
ax_big.plot(
    *binned_radial_profile(model_psf_bin_image, 1, x_cen_snap, y_cen_snap, 0.25),
    label="Model Convolved with PSF Plus Background",
    c=bpl.color_cycle[0],
    lw=4,
)
ax_big.scatter(
    *radial_profile(data_snapshot, 1, x_cen_snap, y_cen_snap),
    label="Data",
    c=bpl.color_cycle[2],
)

# Use the final model to normalize the psf
psf *= np.max(model_psf_bin_image) / np.max(psf)
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
y_at_reff = profile_at_radius(r_eff)
ax_big.plot([r_eff, r_eff], [0, y_at_reff], c=bpl.almost_black, ls="--")
ax_big.add_text(
    x=r_eff - 0.05,
    y=15,
    text="$R_{eff}$",
    ha="right",
    va="bottom",
    rotation=90,
    fontsize=25,
)

y_at_rmax = profile_at_radius(15)
ax_big.plot([15, 15], [0, y_at_rmax], c=bpl.almost_black, ls="--")
ax_big.add_text(
    x=15 - 0.05,
    y=y_at_rmax * 0.8,
    text="$R_{max}$",
    ha="right",
    va="top",
    rotation=90,
    fontsize=25,
)

ax_big.legend()
ax_big.set_yscale("log")
ax_big.add_labels("Radius [pixels]", "Pixel Value [e$^-$]")
max_pix = 15
ax_big.set_limits(
    0, max_pix, 5, 1000,
)

# then add a second scale on top translating into parsecs
max_pc, _, _ = utils.pixels_to_pc_with_errors(data_dir, max_pix, 0, 0)
ax_big.twin_axis_simple("x", lower_lim=0, upper_lim=max_pc, label="Radius [pc]")

fig.savefig(plot_name)
