import numpy as np
from scipy import optimize
from matplotlib import colors
import betterplotlib as bpl


bpl.set_style()
# ======================================================================================
#
# functions for writing to an output file with fit info
#
# ======================================================================================
def write_fit_results(fit_out_file, name, number, best_fit_params, fit_history):
    """
    Write the results of one fit to a file, which can be copied into
    :param fit_out_file:
    :param name:
    :param number:
    :param best_fit_params:
    :param fit_history:
    :return:
    """
    print_str = f"\t\t{name} & {number}"
    # the second parameter is the log of clusters at 10^4. Put it back to linear space
    best_fit_params[1] = 10 ** best_fit_params[1]
    fit_history[1] = [10 ** f for f in fit_history[1]]
    for idx in range(len(best_fit_params)):
        std = np.std(fit_history[idx])
        print_str += f" & {best_fit_params[idx]:.3f} $\pm$ {std:.3f}"
    print_str += "\\\\ \n"
    fit_out_file.write(print_str)


def out_file_spacer(fit_out_file):
    fit_out_file.write("\t\t\midrule\n")


# ======================================================================================
#
# Simple fitting of the mass-size model
#
# ======================================================================================
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

    :param params: Slope, height at the pivot point, and standard deviation of
                   intrinsic scatter
    :return: Value for the negative log likelihood
    """
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = params[1] - params[0] * pivot_point_x

    data_variance = project_data_variance(
        xs, x_err_down, x_err_up, ys, y_err_down, y_err_up, params[0], intercept
    )
    data_diffs = project_data_differences(xs, ys, params[0], intercept)

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


def fit_mass_size_relation(
    log_mass,
    log_mass_err_lo,
    log_mass_err_hi,
    log_r_eff,
    log_r_eff_err_lo,
    log_r_eff_err_hi,
    fit_mass_lower_limit=1e-5,
    fit_mass_upper_limit=1e10,
):
    fit_mask = log_mass > np.log10(fit_mass_lower_limit)
    fit_mask = np.logical_and(fit_mask, log_mass < np.log10(fit_mass_upper_limit))

    log_mass = log_mass[fit_mask]
    log_mass_err_lo = log_mass_err_lo[fit_mask]
    log_mass_err_hi = log_mass_err_hi[fit_mask]
    log_r_eff = log_r_eff[fit_mask]
    log_r_eff_err_lo = log_r_eff_err_lo[fit_mask]
    log_r_eff_err_hi = log_r_eff_err_hi[fit_mask]

    # set some of the convergence criteria parameters for the Powell fitting routine.
    xtol = 1e-10
    ftol = 1e-10
    maxfev = np.inf
    maxiter = np.inf
    # Then try the fitting
    best_fit_result = optimize.minimize(
        negative_log_likelihood,
        args=(
            log_mass,
            log_mass_err_lo,
            log_mass_err_hi,
            log_r_eff,
            log_r_eff_err_lo,
            log_r_eff_err_hi,
        ),
        bounds=([-1, 1], [None, None], [0, 0.5]),
        x0=np.array([0.2, np.log10(2), 0.3]),
        method="Powell",
        options={
            "xtol": xtol,
            "ftol": ftol,
            "maxfev": maxfev,
            "maxiter": maxiter,
        },
    )
    assert best_fit_result.success

    # best_slope, best_intercept, best_intrinsic_scatter = best_fit_result.x

    # Then do bootstrapping
    n_variables = len(best_fit_result.x)
    param_history = [[] for _ in range(n_variables)]
    param_std_last = [np.inf for _ in range(n_variables)]

    converge_criteria = 0.1  # fractional change in std required for convergence
    converged = [False for _ in range(n_variables)]
    check_spacing = 10  # how many iterations between checking the std
    iteration = 0
    while not all(converged):
        iteration += 1

        # create a new sample of x and y coordinates
        sample_idxs = np.random.randint(0, len(log_mass), len(log_mass))

        # fit to this set of data
        this_result = optimize.minimize(
            negative_log_likelihood,
            args=(
                log_mass[sample_idxs],
                log_mass_err_lo[sample_idxs],
                log_mass_err_hi[sample_idxs],
                log_r_eff[sample_idxs],
                log_r_eff_err_lo[sample_idxs],
                log_r_eff_err_hi[sample_idxs],
            ),
            bounds=([-1, 1], [None, None], [0, 0.5]),
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

    return best_fit_result.x, param_history


# ======================================================================================
#
# Plotting functionality - percentiles
#
# ======================================================================================
def get_r_percentiles(radii, masses, percentile, d_log_M):
    bins = np.logspace(0, 7, int(5 / d_log_M) + 1)

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
        if len(good_radii) > 20:
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
    for left_idx in range(0, len(radii) - dn, dn):
        right_idx = left_idx + n
        # fix the last bin
        if right_idx > len(idxs_sort):
            right_idx = len(idxs_sort)
            left_idx = right_idx - n

        idxs = idxs_sort[left_idx:right_idx]
        this_masses = masses[idxs]
        this_radii = radii[idxs]

        masses_median.append(np.median(this_masses))
        radii_percentiles.append(np.percentile(this_radii, percentile))
    return masses_median, radii_percentiles


def get_r_percentiles_unique_values(radii, ages, percentile):
    # get the unique ages
    unique_ages = np.unique(ages)
    # cut off values above 1e9
    unique_ages = unique_ages[unique_ages <= 1e9]
    radii_percentiles = []
    for age in unique_ages:
        mask = ages == age
        radii_percentiles.append(np.percentile(radii[mask], percentile))
    return unique_ages, radii_percentiles


def add_percentile_lines(ax, mass, r_eff, style="moving", color=bpl.almost_black):
    # plot the median and the IQR
    for percentile in [5, 25, 50, 75, 95]:
        if style == "moving":
            mass_bins, radii_percentile = get_r_percentiles_moving(
                r_eff, mass, percentile, 200, 200
            )
        elif style == "unique":
            mass_bins, radii_percentile = get_r_percentiles_unique_values(
                r_eff, mass, percentile
            )
        elif style == "fixed_width":
            mass_bins, radii_percentile = get_r_percentiles(
                r_eff, mass, percentile, 0.1
            )
        else:
            raise ValueError("Style not recognized")
        ax.plot(
            mass_bins,
            radii_percentile,
            c=color,
            lw=3 * (1 - (abs(percentile - 50) / 50)) + 0.5,
            zorder=9,
        )
        ax.text(
            x=mass_bins[0],
            y=radii_percentile[0],
            ha="right",
            va="center",
            s=percentile,
            fontsize=16,
            zorder=100,
        )


# ======================================================================================
#
# Plotting functionality - psfs
#
# ======================================================================================
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


def add_psfs_to_plot(ax, x_max=1e6):
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
        ax.plot(
            [0.7 * x_max, x_max],
            [psf_size_pc, psf_size_pc],
            lw=1,
            c=bpl.almost_black,
            zorder=3,
        )


# ======================================================================================
#
# Plotting functionality - fit lines
#
# ======================================================================================
def plot_best_fit_line(
    ax,
    best_fit_params,
    fit_mass_lower_limit=1,
    fit_mass_upper_limit=1e6,
    color=bpl.color_cycle[1],
    fill=True,
    label=None,
):
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = best_fit_params[1] - best_fit_params[0] * pivot_point_x

    # Make the string that will be used for the label
    if label is None:
        label = "$R_{eff} = "
        label += f"{10**best_fit_params[1]:.2f}"
        label += "\left( \\frac{M}{10^4 M_\odot} \\right)^{"
        label += f"{best_fit_params[0]:.2f}"
        label += "}$"

    plot_log_masses = np.arange(
        np.log10(fit_mass_lower_limit), np.log10(fit_mass_upper_limit), 0.01
    )
    plot_log_radii = best_fit_params[0] * plot_log_masses + intercept
    ax.plot(
        10 ** plot_log_masses,
        10 ** plot_log_radii,
        c=color,
        lw=4,
        zorder=8,
        label=label,
    )
    if fill:
        ax.fill_between(
            x=10 ** plot_log_masses,
            y1=10 ** (plot_log_radii - best_fit_params[2]),
            y2=10 ** (plot_log_radii + best_fit_params[2]),
            color=color,
            alpha=0.5,
            zorder=0,
            label="$\sigma_{int}$" + f" = {best_fit_params[2]:.2f} dex",
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


# ======================================================================================
#
# Plotting functionality - datasets
#
# ======================================================================================
def plot_mass_size_dataset_scatter(
    ax,
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    color,
    label=None,
):
    ax.scatter(mass, r_eff, alpha=1.0, s=3, c=color, zorder=4, label=label)
    # Have errorbars separately so they can easily be turned off
    ax.errorbar(
        x=mass,
        y=r_eff,
        alpha=1.0,
        markersize=0,
        yerr=[r_eff_err_lo, r_eff_err_hi],
        xerr=[mass_err_lo, mass_err_hi],
        lw=0.1,
        zorder=3,
        c=color,
    )


def create_color_cmap(hex_color, min_saturation=0.1, max_value=0.8):
    """
    Create a colormap that fades from one color to nearly white.

    This is done by converting the color to HSV, then decreasing the saturation while
    increasing the value (which makes it closer to white)

    :param hex_color: Original starting color, must be in hex format
    :param min_saturation: The saturation of the point farthest from the original color
    :param max_value: The value of the point farthest from the original color
    :return: A matplotilb colormap. Calling it with 0 returns the color specififed
             by `min_saturation` and `max_value` while keeping the same hue, while
             1 will return the original color.
    """
    # convert to HSV (rgb required as an intermediate)
    base_color_rgb = colors.hex2color(hex_color)
    h, s, v = colors.rgb_to_hsv(base_color_rgb)
    N = 256  # number of points in final colormap
    # reduce the saturation and up the brightness. Start from the outer values, as these
    # will correspond to 0, while the original color will be 1
    saturations = np.linspace(min_saturation, s, N)
    values = np.linspace(max_value, v, N)
    out_xs = np.linspace(0, 1, N)

    # set up the weird format required by LinearSegmentedColormap
    cmap_dict = {"red": [], "blue": [], "green": []}
    for idx in range(N):
        r, g, b = colors.hsv_to_rgb((h, saturations[idx], values[idx]))
        out_x = out_xs[idx]
        # LinearSegmentedColormap requires a weird format. I don't think the difference
        # in the last two values matters, it seems to work fine without it.
        cmap_dict["red"].append((out_x, r, r))
        cmap_dict["green"].append((out_x, g, g))
        cmap_dict["blue"].append((out_x, b, b))
    return colors.LinearSegmentedColormap(hex_color, cmap_dict, N=256)


def plot_mass_size_dataset_contour(ax, mass, r_eff, color, zorder=5):
    cmap = create_color_cmap(color)
    # use median errors as the smoothing. First get mean min and max of all clusters,
    # then take the median of that
    # k = 1.25
    # x_smoothing = k * np.median(np.mean([log_r_eff_err_lo, log_r_eff_err_hi], axis=0))
    # y_smoothing = k * np.median(np.mean([log_mass_err_lo, log_mass_err_hi], axis=0))
    x_smoothing = 0.08
    y_smoothing = 0.08

    common = {
        "percent_levels": [0.5, 0.9],
        "smoothing": [x_smoothing, y_smoothing],  # dex
        "bin_size": 0.01,  # dex
        "log": True,
        "cmap": cmap,
    }
    ax.density_contourf(mass, r_eff, alpha=0.6, zorder=zorder, **common)
    ax.density_contour(mass, r_eff, zorder=zorder + 20, **common)


def format_mass_size_plot(ax, xmin=1e2, xmax=1e6):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_limits(xmin, xmax, 0.1, 40)
    ax.add_labels("Cluster Mass [M$_\odot$]", "Cluster Effective Radius [pc]")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.legend(loc=2, frameon=False)
