"""
mass_size.py - plot the mass-size relation for LEGUS clusters
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits
from scipy import optimize
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

mask = np.logical_and(mask, big_catalog["good"])
print(f"Clusters with good fits: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["mass_msun_max"] > 1)
print(f"Clusters with nonzero mass error: {np.sum(mask)}")

# ======================================================================================
#
# Fit the mass-size model
#
# ======================================================================================
fit_mass_lower_limit = 1e3
fit_mass_upper_limit = np.inf
fit_mask = np.logical_and(mask, big_catalog["mass_msun"] > fit_mass_lower_limit)
fit_mask = np.logical_and(fit_mask, big_catalog["mass_msun"] < fit_mass_upper_limit)

# First get the parameters to be used, and transform them into log
log_mass = np.log10(big_catalog["mass_msun"][fit_mask])
# mass errors are reported as min and max values
log_mass_err_hi = np.log10(big_catalog["mass_msun_max"][fit_mask]) - log_mass
log_mass_err_lo = log_mass - np.log10(big_catalog["mass_msun_min"][fit_mask])

# do the same thing with the radii, although it's a bit uglier since we don't report
# min and max, just the errors
r_eff = big_catalog["r_eff_pc_rmax_15pix_best"][fit_mask]
r_eff_err_hi = big_catalog["r_eff_pc_rmax_15pix_e+"][fit_mask]
r_eff_err_lo = big_catalog["r_eff_pc_rmax_15pix_e-"][fit_mask]

log_r_eff = np.log10(r_eff)
log_r_eff_err_hi = np.log10(r_eff + r_eff_err_hi) - log_r_eff
log_r_eff_err_lo = log_r_eff - np.log10(r_eff - r_eff_err_lo)

# This fitting is based off the prescriptions in Hogg, Bovy, Lang 2010 (arxiv:1008.4686)
# sections 7 and 8. We incorporate the uncertainties in x and y by calculating the
# difference and sigma along the line orthogonal to the best fit line.

# define the functions that project the distance and uncertainty along the line
def unit_vector_perp_to_line(slope):
    """
    Equation 29 of Hogg, Bovy, Lang 2010 (arxiv:1008.4686)
    :param slope: Slope of the line
    :return: Two component unit vector perpendicular to the line
    """
    return np.array([-slope, 1]).T / np.sqrt(1 + slope ** 2)


def project_data_differences(slope, intercept):
    """
    Calculate the orthogonal displacement of all data points from the line specified.
    See equation 30 of Hogg, Bovy, Lang 2010 (arxiv:1008.4686)

    I made a simpler version of this equation by examining the geometry of the
    situation. The orthogonal direction to a line can be obtained fairly easily.

    I include a commented out version of the original implementation. Both
    implementations give the same results

    :param slope: Slope of the line
    :param intercept: Intercept of the line
    :return: Orthogonal displacement of all datapoints from this line
    """
    # v_hat = unit_vector_perp_to_line(slope)
    # data_array = np.array([masses_log, r_eff_log])
    # dot_term = np.dot(v_hat, data_array)
    #
    # theta = np.arctan(slope)
    # return dot_term - intercept * np.cos(theta)

    return np.cos(np.arctan(slope)) * (log_r_eff - (slope * log_mass + intercept))


def project_data_variance(slope, intercept):
    """
    Calculate the orthogonal uncertainty of all data points from the line specified.
    See equation 31 of Hogg, Bovy, Lang 2010 (arxiv:1008.4686)

    :param slope: Slope of the line
    :param intercept: Intercept of the line
    :return: Orthogonal displacement of all datapoints from this line
    """
    # make dummy error arrays, which will be filled later
    log_r_eff_errors = np.zeros(log_r_eff_err_lo.shape)
    log_mass_errors = np.zeros(log_mass_err_lo.shape)
    # determine which errors to use. This is done on a datapoint by datapoint basis.
    # If the datapoint is above the best fit line, we use the lower errors on reff,
    # if it is below the line we use the upper errors.
    expected_values = log_mass * slope + intercept
    mask_above = log_r_eff > expected_values

    log_r_eff_errors[mask_above] = log_r_eff_err_lo[mask_above]
    log_r_eff_errors[~mask_above] = log_r_eff_err_hi[~mask_above]

    # Errors on mass are similar, but it depends on the sign of the slope. For a
    # positive slope, we use the upper errors for points above the line, and lower
    # errors for points below the line. This is opposite for negative slope. This can
    # be determined by examining the direction the orthogonal line will go in each of
    # these cases.
    if slope > 0:
        log_mass_errors[mask_above] = log_mass_err_hi[mask_above]
        log_mass_errors[~mask_above] = log_mass_err_lo[~mask_above]
    else:
        log_mass_errors[mask_above] = log_mass_err_lo[mask_above]
        log_mass_errors[~mask_above] = log_mass_err_hi[~mask_above]
    # convert to variance
    log_mass_variance = log_mass_errors ** 2
    log_r_eff_variance = log_r_eff_errors ** 2

    # Then we follow the equation 31 to project this along the direction requested.
    # Since our covariance array is diagonal already (no covariance terms), Equation
    # 26 is simple and Equation 31 can be simplified. Note that this has limits
    # of log_mass_variance if slope = infinity (makes sense, as the horizontal direction
    # would be perpendicular to that line), and log_r_eff_variance if slope = 0 (makes
    # sense, as the vertical direction is perpendicular to that line).
    return (slope ** 2 * log_mass_variance + log_r_eff_variance) / (1 + slope ** 2)


# Then we can define the functions to minimize
def negative_log_likelihood(params):
    """
    Function to be minimized. We use negative log likelihood, as minimizing this
    maximizes likelihood.

    The functional form is taken from Hogg, Bovy, Lang 2010 (arxiv:1008.4686) eq 35.

    :param params: Slope, intercept, and standard deviation of intrinsic scatter
    :return: Value for the negative log likelihood
    """
    data_variance = project_data_variance(params[0], params[1])
    data_diffs = project_data_differences(params[0], params[1])

    # calculate the sum of data likelihoods
    data_likelihoods = -0.5 * np.sum(
        (data_diffs ** 2) / (data_variance + params[2] ** 2)
    )
    # then penalize large intrinsic scatter
    scatter_likelihood = -0.5 * np.sum(np.log(data_variance + params[2] ** 2))
    # up to a constant, the sum of these is the likelihood. Return the negative of it
    # to get the negative log likelihood
    return -1 * (data_likelihoods + scatter_likelihood)


# set some of the convergence criteria parameters for the Powell fitting routine.
xtol = 1e-10
ftol = 1e-10
maxfev = np.inf
maxiter = np.inf
# Then try the fitting
fit_result = optimize.minimize(
    negative_log_likelihood,
    x0=[0.1, 0, 0],
    method="Powell",
    options={
        "xtol": xtol,
        "ftol": ftol,
        "maxfev": maxfev,
        "maxiter": maxiter,
    },
)
assert fit_result.success
print(fit_result.x)

best_slope, best_intercept, best_intrinsic_scatter = fit_result.x

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
        # and plot the best fit line
        plot_log_masses = np.arange(0, 10, 0.01)
        plot_log_radii = best_slope * plot_log_masses + best_intercept
        ax.plot(
            10 ** plot_log_masses,
            10 ** plot_log_radii,
            c=bpl.color_cycle[1],
            lw=4,
            zorder=0,
        )
        ax.plot(
            10 ** plot_log_masses,
            10 ** (plot_log_radii + best_intrinsic_scatter),
            c=bpl.color_cycle[1],
            lw=2,
            zorder=0,
        )
        ax.plot(
            10 ** plot_log_masses,
            10 ** (plot_log_radii - best_intrinsic_scatter),
            c=bpl.color_cycle[1],
            lw=2,
            zorder=0,
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
    ax.set_limits(1e2, 1e6, 0.1, 40)
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
    all_ratios,
    alpha=1.0,
    lw=1,
    color=bpl.color_cycle[3],
    bins=np.logspace(-1, 1, 21),
)

ax.axvline(1.0)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_limits(0.1, 10)
ax.add_labels("Cluster Effective Radius / PSF Effective Radius", "Number of Clusters")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
fig.savefig(plot_name.parent / "r_eff_over_psf.png", bbox_inches="tight")
