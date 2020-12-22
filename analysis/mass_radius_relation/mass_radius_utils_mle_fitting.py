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


def project_data_variance(x_err, y_err, slope):
    """
    Calculate the orthogonal uncertainty of all data points from the line specified.
    See equation 31 of Hogg, Bovy, Lang 2010 (arxiv:1008.4686)

    :param slope: Slope of the line
    :param intercept: Intercept of the line
    :return: Orthogonal displacement of all datapoints from this line
    """
    # We follow the equation 31 to project this along the direction requested.
    # Since our covariance array is diagonal already (no covariance terms), Equation
    # 26 is simple and Equation 31 can be simplified. Note that this has limits
    # of x_variance if slope = infinity (makes sense, as the horizontal direction
    # would be perpendicular to that line), and y_variance if slope = 0 (makes
    # sense, as the vertical direction is perpendicular to that line).
    return (slope ** 2 * x_err ** 2 + y_err ** 2) / (1 + slope ** 2)


# Then we can define the functions to minimize
def negative_log_likelihood_orthogonal(params, xs, x_err, ys, y_err):
    """
    Function to be minimized. We use negative log likelihood, as minimizing this
    maximizes likelihood.

    The functional form is taken from Hogg, Bovy, Lang 2010 (arxiv:1008.4686) eq 35.
    This takes the difference orthogonal to the best fit line

    :param params: Slope, height at the pivot point, and standard deviation of
                   intrinsic scatter
    :return: Value for the negative log likelihood
    """
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = params[1] - params[0] * pivot_point_x

    # variance is the same as the orthogonal fit, as we account for errors in the
    # x and y directions
    data_variance = project_data_variance(x_err, y_err, params[0])
    total_variance = data_variance + params[2] ** 2
    # calculate the difference orthogonal to the best fit line
    data_diffs = project_data_differences(xs, ys, params[0], intercept)

    # calculate the sum of data likelihoods
    data_likelihoods = -0.5 * np.sum((data_diffs ** 2) / total_variance)

    # then penalize large intrinsic scatter. This term really comes from the definition
    # of a Gaussian likelihood. This term is always out front of a Gaussian, but
    # normally it's just a constant. When we include intrinsic scatter it now
    # affects the likelihood.
    scatter_likelihood = -0.5 * np.sum(np.log(total_variance))
    # up to a constant, the sum of these is the likelihood. Return the negative of it
    # to get the negative log likelihood
    return -1 * (data_likelihoods + scatter_likelihood)


def negative_log_likelihood_vertical(params, xs, x_err, ys, y_err):
    """
    Function to be minimized. We use negative log likelihood, as minimizing this
    maximizes likelihood.

    This accounts for x and y errors, but takes the vertial difference from the best
    fit line

    :param params: Slope, height at the pivot point, and standard deviation of
                   intrinsic scatter
    :return: Value for the negative log likelihood
    """
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = params[1] - params[0] * pivot_point_x

    # variance is the same as the orthogonal fit, as we account for errors in the
    # x and y directions
    data_variance = project_data_variance(x_err, y_err, params[0])
    total_variance = data_variance + params[2] ** 2
    # the differences are simply the differences in y
    expected_ys = intercept + params[0] * xs
    data_diffs = ys - expected_ys

    # calculate the sum of data likelihoods
    data_likelihoods = -0.5 * np.sum((data_diffs ** 2) / total_variance)

    # then penalize large intrinsic scatter. This term really comes from the definition
    # of a Gaussian likelihood. This term is always out front of a Gaussian, but
    # normally it's just a constant. When we include intrinsic scatter it now
    # affects the likelihood.
    scatter_likelihood = -0.5 * np.sum(np.log(total_variance))
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
    fit_style="orthogonal",
):
    # choose the likelihood function to use
    if fit_style == "orthogonal":
        negative_log_likelihood = negative_log_likelihood_orthogonal
    elif fit_style == "vertical":
        negative_log_likelihood = negative_log_likelihood_vertical
    else:
        raise ValueError("fit style not recognized")

    log_mass, log_mass_err_lo, log_mass_err_hi = mru.transform_to_log(
        mass, mass_err_lo, mass_err_hi
    )
    log_r_eff, log_r_eff_err_lo, log_r_eff_err_hi = mru.transform_to_log(
        r_eff, r_eff_err_lo, r_eff_err_hi
    )
    # and symmetrize the errors
    log_mass_err = np.mean([log_mass_err_lo, log_mass_err_hi], axis=0)
    log_r_eff_err = np.mean([log_r_eff_err_lo, log_r_eff_err_hi], axis=0)

    fit_mask = log_mass > np.log10(fit_mass_lower_limit)
    fit_mask = np.logical_and(fit_mask, log_mass < np.log10(fit_mass_upper_limit))

    log_mass = log_mass[fit_mask]
    log_mass_err = log_mass_err[fit_mask]
    log_r_eff = log_r_eff[fit_mask]
    log_r_eff_err = log_r_eff_err[fit_mask]

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
            log_mass_err,
            log_r_eff,
            log_r_eff_err,
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
                log_mass_err[sample_idxs],
                log_r_eff[sample_idxs],
                log_r_eff_err[sample_idxs],
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