import numpy as np
from scipy import optimize

import mass_radius_utils as mru

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
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    fit_mass_lower_limit=1e-5,
    fit_mass_upper_limit=1e10,
):
    log_mass, log_mass_err_lo, log_mass_err_hi = mru.transform_to_log(
        mass, mass_err_lo, mass_err_hi
    )
    log_r_eff, log_r_eff_err_lo, log_r_eff_err_hi = mru.transform_to_log(
        r_eff, r_eff_err_lo, r_eff_err_hi
    )

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
