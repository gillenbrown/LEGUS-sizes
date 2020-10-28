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

mass_plot = big_catalog["mass_msun"][mask]
# mass errors are reported as min and max values
mass_err_hi_plot = big_catalog["mass_msun_max"][mask] - mass_plot
mass_err_lo_plot = mass_plot - big_catalog["mass_msun_min"][mask]

r_eff_plot = big_catalog["r_eff_pc_rmax_15pix_best"][mask]
r_eff_err_hi_plot = big_catalog["r_eff_pc_rmax_15pix_e+"][mask]
r_eff_err_lo_plot = big_catalog["r_eff_pc_rmax_15pix_e-"][mask]


# ======================================================================================
#
# Fit the mass-size model
#
# ======================================================================================
fit_mass_lower_limit = 1e1
fit_mass_upper_limit = 1e6
fit_mask = np.logical_and(mask, big_catalog["mass_msun"] > fit_mass_lower_limit)
fit_mask = np.logical_and(fit_mask, big_catalog["mass_msun"] < fit_mass_upper_limit)

# First get the parameters to be used, and transform them into log
log_mass_fit = np.log10(big_catalog["mass_msun"][fit_mask])
# mass errors are reported as min and max values
log_mass_err_hi_fit = np.log10(big_catalog["mass_msun_max"][fit_mask]) - log_mass_fit
log_mass_err_lo_fit = log_mass_fit - np.log10(big_catalog["mass_msun_min"][fit_mask])

# do the same thing with the radii, although it's a bit uglier since we don't report
# min and max, just the errors
r_eff_fit = big_catalog["r_eff_pc_rmax_15pix_best"][fit_mask]
r_eff_err_hi_fit = big_catalog["r_eff_pc_rmax_15pix_e+"][fit_mask]
r_eff_err_lo_fit = big_catalog["r_eff_pc_rmax_15pix_e-"][fit_mask]

log_r_eff_fit = np.log10(r_eff_fit)
log_r_eff_err_hi_fit = np.log10(r_eff_fit + r_eff_err_hi_fit) - log_r_eff_fit
log_r_eff_err_lo_fit = log_r_eff_fit - np.log10(r_eff_fit - r_eff_err_lo_fit)

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


def project_data_differences(xs, ys, slope, intercept):
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

    return np.cos(np.arctan(slope)) * (ys - (slope * xs + intercept))


def project_data_variance(
    xs, x_err_down, x_err_up, ys, y_err_down, y_err_up, slope, intercept
):
    """
    Calculate the orthogonal uncertainty of all data points from the line specified.
    See equation 31 of Hogg, Bovy, Lang 2010 (arxiv:1008.4686)

    :param slope: Slope of the line
    :param intercept: Intercept of the line
    :return: Orthogonal displacement of all datapoints from this line
    """
    # make dummy error arrays, which will be filled later
    x_errors = np.zeros(xs.shape)
    y_errors = np.zeros(ys.shape)
    # determine which errors to use. This is done on a datapoint by datapoint basis.
    # If the datapoint is above the best fit line, we use the lower errors on y,
    # if it is below the line we use the upper errors.
    expected_values = xs * slope + intercept
    mask_above = ys > expected_values

    y_errors[mask_above] = y_err_down[mask_above]
    y_errors[~mask_above] = y_err_up[~mask_above]

    # Errors on x are similar, but it depends on the sign of the slope. For a
    # positive slope, we use the upper errors for points above the line, and lower
    # errors for points below the line. This is opposite for negative slope. This can
    # be determined by examining the direction the orthogonal line will go in each of
    # these cases.
    if slope > 0:
        x_errors[mask_above] = x_err_up[mask_above]
        x_errors[~mask_above] = x_err_down[~mask_above]
    else:
        x_errors[mask_above] = x_err_down[mask_above]
        x_errors[~mask_above] = x_err_up[~mask_above]
    # convert to variance
    x_variance = x_errors ** 2
    y_variance = y_errors ** 2

    # Then we follow the equation 31 to project this along the direction requested.
    # Since our covariance array is diagonal already (no covariance terms), Equation
    # 26 is simple and Equation 31 can be simplified. Note that this has limits
    # of x_variance if slope = infinity (makes sense, as the horizontal direction
    # would be perpendicular to that line), and y_variance if slope = 0 (makes
    # sense, as the vertical direction is perpendicular to that line).
    return (slope ** 2 * x_variance + y_variance) / (1 + slope ** 2)


# Then we can define the functions to minimize
def negative_log_likelihood(params, xs, x_err_down, x_err_up, ys, y_err_down, y_err_up):
    """
    Function to be minimized. We use negative log likelihood, as minimizing this
    maximizes likelihood.

    The functional form is taken from Hogg, Bovy, Lang 2010 (arxiv:1008.4686) eq 35.

    :param params: Slope, intercept, and standard deviation of intrinsic scatter
    :return: Value for the negative log likelihood
    """
    data_variance = project_data_variance(
        xs, x_err_down, x_err_up, ys, y_err_down, y_err_up, params[0], params[1]
    )
    data_diffs = project_data_differences(xs, ys, params[0], params[1])

    # calculate the sum of data likelihoods
    data_likelihoods = -0.5 * np.sum(
        (data_diffs ** 2) / (data_variance + params[2] ** 2)
    )
    # then penalize large intrinsic scatter. This term really comes from the definition
    # of a Gaussian likelihood. This term is always out front of a Gaussian, but
    # normally it's just a constant. When we include intrinsic scatter it now
    # affects the likelihood.
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
best_fit_result = optimize.minimize(
    negative_log_likelihood,
    args=(
        log_mass_fit,
        log_mass_err_lo_fit,
        log_mass_err_hi_fit,
        log_r_eff_fit,
        log_r_eff_err_lo_fit,
        log_r_eff_err_hi_fit,
    ),
    bounds=([None, None], [None, None], [0, None]),
    x0=np.array([0.2, 0, 0.3]),
    method="Powell",
    options={
        "xtol": xtol,
        "ftol": ftol,
        "maxfev": maxfev,
        "maxiter": maxiter,
    },
)
assert best_fit_result.success
print(best_fit_result.x)

best_slope, best_intercept, best_intrinsic_scatter = best_fit_result.x

# Then do bootstrapping
n_variables = len(best_fit_result.x)
param_history = [[] for _ in range(n_variables)]
param_std_last = [np.inf for _ in range(n_variables)]

converge_criteria = 0.1  # fractional change in std required for convergence
converged = [False for _ in range(3)]
check_spacing = 2  # how many iterations between checking the std
iteration = 0
while not all(converged):
    iteration += 1

    # create a new sample of x and y coordinates
    sample_idxs = np.random.randint(0, len(log_mass_fit), len(log_mass_fit))

    # fit to this set of data
    this_result = optimize.minimize(
        negative_log_likelihood,
        args=(
            log_mass_fit[sample_idxs],
            log_mass_err_lo_fit[sample_idxs],
            log_mass_err_hi_fit[sample_idxs],
            log_r_eff_fit[sample_idxs],
            log_r_eff_err_lo_fit[sample_idxs],
            log_r_eff_err_hi_fit[sample_idxs],
        ),
        bounds=([None, None], [None, None], [0, None]),
        x0=best_fit_result.x,
        method="Powell",
        options={
            "xtol": xtol,
            "ftol": ftol,
            "maxfev": maxfev,
            "maxiter": maxiter,
        },
    )

    assert this_result.success
    # store the results
    for param_idx in range(n_variables):
        param_history[param_idx].append(this_result.x[param_idx])

    # then check if we're converged
    if iteration % check_spacing == 0:
        for param_idx in range(n_variables):
            # calculate the new standard deviation
            this_std = np.std(param_history[param_idx])
            if this_std == 0:
                converged[param_idx] = True
            else:  # actually calculate the change
                last_std = param_std_last[param_idx]
                diff = abs((this_std - last_std) / this_std)
                converged[param_idx] = diff < converge_criteria

            # then set the new last value
            param_std_last[param_idx] = this_std

print(param_std_last)
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


def get_r_percentiles_moving(radii, masses, percentile, n, dn):
    # go through the masses in sorted order
    idxs_sort = np.argsort(masses)
    # then go through chunks of them at a time to get the medians
    masses_median = []
    radii_percentiles = []
    for left_idx in range(0, len(radii) - n, dn):
        right_idx = left_idx + n
        idxs = idxs_sort[left_idx:right_idx]
        this_masses = masses[idxs]
        this_radii = radii[idxs]

        masses_median.append(np.median(this_masses))
        radii_percentiles.append(np.percentile(this_radii, percentile))
    return masses_median, radii_percentiles


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
fig, ax = bpl.subplots()

ax.scatter(mass_plot, r_eff_plot, alpha=1.0, s=3, c=bpl.color_cycle[0], zorder=4)
# Have errorbars separately so they can easily be turned off
ax.errorbar(
    x=mass_plot,
    y=r_eff_plot,
    alpha=1.0,
    markersize=0,
    yerr=[r_eff_err_lo_plot, r_eff_err_hi_plot],
    xerr=[mass_err_lo_plot, mass_err_hi_plot],
    lw=0.1,
    zorder=3,
    c=bpl.color_cycle[0],
)
# plot the median and the IQR
d_log_M = 0.25
for percentile in [5, 10, 25, 50, 75, 90, 95]:
    # mass_bins, radii_percentile = get_r_percentiles(
    #     r_eff_plot, mass_plot, percentile, d_log_M
    # )
    mass_bins, radii_percentile = get_r_percentiles_moving(
        r_eff_plot, mass_plot, percentile, 100, 50
    )
    ax.plot(
        mass_bins,
        radii_percentile,
        c=bpl.almost_black,
        lw=2 * (1 - (abs(percentile - 50) / 50)) + 0.5,
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
plot_log_masses = np.arange(
    np.log10(fit_mass_lower_limit), np.log10(fit_mass_upper_limit), 0.01
)
plot_log_radii = best_slope * plot_log_masses + best_intercept
ax.plot(
    10 ** plot_log_masses,
    10 ** plot_log_radii,
    c=bpl.color_cycle[3],
    lw=4,
    zorder=10,
    label="$R_{eff} \propto M^{" + f"{best_slope:.2f}" + "}$",
)
ax.fill_between(
    x=10 ** plot_log_masses,
    y1=10 ** (plot_log_radii - best_intrinsic_scatter),
    y2=10 ** (plot_log_radii + best_intrinsic_scatter),
    color="0.8",
    zorder=0,
    label=f"Intrinsic Scatter = {best_intrinsic_scatter:.2f} dex",
)


# Filled in bootstrap interval is currently turned off because the itnerval is smaller
# than the width of the line
# # Then add the shaded region of regions allowed by bootstrapping. We'll calculate
# # the fit line for all the iterations, then at each x value calculate the 68
# # percent range to shade between.
# lines = [[] for _ in range(len(plot_log_masses))]
# for i in range(len(param_history[0])):
#     this_line = param_history[0][i] * plot_log_masses + param_history[1][i]
#     for j in range(len(this_line)):
#         lines[j].append(this_line[j])
# # Then we can calculate the percentile at each location. The y is in log here,
# # so scale it back up to regular values
# upper_limits = [10 ** np.percentile(ys, 84.15) for ys in lines]
# lower_limits = [10 ** np.percentile(ys, 15.85) for ys in lines]
#
# ax.fill_between(
#     x=10 ** plot_log_masses,
#     y1=lower_limits,
#     y2=upper_limits,
#     zorder=0,
#     alpha=0.5,
# )

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
    psf_size_arcsec = utils.pixels_to_arcsec(psf_size_pixels, home_dir)
    psf_size_pc = utils.arcsec_to_pc_with_errors(
        home_dir, psf_size_arcsec, 0, 0, False
    )[0]
    ax.plot([7e5, 1e6], [psf_size_pc, psf_size_pc], lw=1, c=bpl.almost_black, zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_limits(1e2, 1e6, 0.1, 40)
ax.add_labels("Cluster Mass [M$_\odot$]", "Cluster Effective Radius [pc]")
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.legend(loc=2)

fig.savefig(plot_name)


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
