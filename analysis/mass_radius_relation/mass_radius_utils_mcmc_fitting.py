from pathlib import Path
import numpy as np
import emcee
from astropy import table
from scipy import interpolate, integrate, special
from tqdm import tqdm

import mass_radius_utils as mru
import corner
import betterplotlib as bpl

bpl.set_style()

# ======================================================================================
#
# SPS models to use in the selection effects calculation
#
# ======================================================================================
# initialize the Yggdrasil table to read the SPS models from, so we can use them later
ygg_name = "Z=0.02_kroupa_IMF_fcov_1_SFR_inst_HST_AB_lyman_alpha0_redshiftgroup3.txt"
ygg_loc = Path(__file__).parent / ygg_name
ygg_table = table.Table.read(ygg_loc, format="ascii")
# restrict to zero redshift
ygg_table = ygg_table[ygg_table["Redshift"] == 0]
# create interpolation objects to get the mass and magnitude at a given age
mass_interp = interpolate.interp1d(
    x=ygg_table["Age(yr)"],
    y=ygg_table["Mstars"],
    bounds_error=False,
    fill_value=(ygg_table["Mstars"][0], ygg_table["Mstars"][-1]),
)
f555_interp = interpolate.interp1d(
    x=ygg_table["Age(yr)"],
    y=ygg_table["F555W"],
    bounds_error=False,
    fill_value=(ygg_table["F555W"][0], ygg_table["F555W"][-1]),
)

# then convenience functions to use this
def get_absolute_mag(mass, age):
    table_mass = mass_interp(age)
    table_mag = f555_interp(age)

    # the correct the magnitude based on the masses.
    return table_mag - 2.5 * np.log10(mass / table_mass)


# ======================================================================================
#
# precalculating V band selection function
#
# ======================================================================================
# I'll also precalculate the selection probability for a grid of mass and age. I do
# this so I don't have to calculate everything each iteration. Note that the V band
# magnitude selection is a step function, which makes the integral analytically solvable
# with a Gaussian likelihood to an error function.
# define the cut for V band absolute magnitude
v_cut = -6


def gaussian_integral_lower(mean, sigma, x_max):
    """
    Integral of a Gaussian function from negative infinity to some value

    This comes from the cumulative distribution function of the Gaussian

    :param mean: Mean of the Gaussian
    :param sigma: Standard deviation of the Gaussian
    :param x_max: Value to integrate to
    :return: Integral from negative infinity to the given value
    """
    return 0.5 * (1 + special.erf((x_max - mean) / (sigma * np.sqrt(2))))


def v_band_selection_probability_raw(log_mass, log_age):
    # The likelihood is Gaussian, with a mean provided by the SPS models. We integrate
    # it up to the cut, which is an error function.
    expected_v = get_absolute_mag(10 ** log_mass, 10 ** log_age)
    v_err = 0.1  # dummy value for now
    return gaussian_integral_lower(expected_v, v_err, v_cut)


# then put this all together in a grid.
min_log_mass, max_log_mass = 0, 10
min_log_age, max_log_age = 3, 12
log_mass_grid = np.arange(min_log_mass, max_log_mass, 0.1)
log_age_grid = np.arange(min_log_age, max_log_age, 0.1)
v_selection_grid = np.zeros((log_mass_grid.size, log_age_grid.size))
print("precalculating V selection function")
for m_idx in tqdm(range(log_mass_grid.size)):
    log_m = log_mass_grid[m_idx]
    for t_idx in range(log_age_grid.size):
        log_t = log_age_grid[t_idx]
        v_selection_grid[m_idx, t_idx] = v_band_selection_probability_raw(log_m, log_t)
# then create the interpolation object. Note that the scipy.interpolate.interp2d notes
# say that the RectBivariateSpline is faster, so that's what I use
v_band_selection_probability = interpolate.RectBivariateSpline(
    x=log_mass_grid, y=log_age_grid, z=v_selection_grid, s=0
)


# ======================================================================================
#
# some helper functions for likelihoods
#
# ======================================================================================
def log_gaussian(x, mean, variance, include_norm=False):
    """
    Natural log of the likelihood for a Gaussian distribution.

    :param x: Value(s) at which to evaluate the likelihood.
    :param mean: Mean of the Gaussian distribution
    :param sigma: Variance of the Gaussian distribution
    :param include_norm: whether or not to calculate the normalization term. This is
                         not needed unless the variance is a parameter of interest
    :return: log of the likelihood at x
    """
    log_likelihood = -((x - mean) ** 2) / (2 * variance)
    if include_norm:
        log_likelihood -= (1 / 2) * np.log(2 * np.pi * variance)
    return log_likelihood


def gaussian(x, mean, variance, include_norm=False):
    """
    Equivalent of log_gaussian, but returns the likelhood, not log of the likelihood
    """
    return np.exp(log_gaussian(x, mean, variance, include_norm))


def mass_size_relation_mean_log(log_mass, beta, log_r_4):
    """
    Return the mean value of the mass-radius relation.

    :param log_mass: Log of the true intrinsic cluster mass
    :param beta: Slope of the mass-radius relation
    :param log_r_4: Normalization, defined as the log of the radius at 10^4 Msun
    :return: log(radius) at the given mass
    """
    pivot_point_mass = 4
    return log_r_4 + (log_mass - pivot_point_mass) * beta


# ======================================================================================
#
# selection functions
#
# ======================================================================================
# define a function for the radius selection.
def r_selection(r_pc):
    """
    Probability of a cluster of a given radius being seleted.

    :param r_pc: radius of the cluster in pc
    :return: Probability of selecting a cluster with this radius
    # TODO: actually define this function based on data
    # TODO: define this function in arcsec, not pc
    """
    return np.minimum(r_pc, 1)


def selection_probability(log_true_mass, log_true_age, beta, log_r_4, sigma):
    """
    Determine Phi, the probability of selecting a cluster of a given true mass and age

    This is elaborated more in the paper. But the idea is to integrate over all values
    of any variables that have selection functions.
    \int f(X) p(X|m, t) dX
    where f(X) is the selection function, and p(X|m, t) is the conditional likelihood
    of selecting this variable.

    Here we have two selection variables: radius and V band magnitude. V band magnitude
    has a step function selection function and a Gaussian likelihood, which mean it
    integrates to an error function. The radius is more complicated, so we need to do
    the integration numerically. The fit parameters are needed for the radius selection
    too, as the determine the likelihood of a cluster of a given mass being above the
    selection threshold.

    :param log_true_mass: log of the true mass of the cluster
    :param log_true_age: log of the true age of the cluster
    :param beta, log_r_4, sigma: Parameters of the mass-radius relation, see
                                 `mass_size_relation_mean_log` for more
    :return: The probability that a cluster of this mass and age will pass the selection
             criteria
    """
    v_term = v_band_selection_probability(log_true_mass, log_true_age)

    def integrand_radius(log_r):
        # here we multiply the selection function times the likelihood. Note that we
        # need the raw likelihood, not the log likelihood
        return r_selection(10 ** log_r) * gaussian(
            log_r, expected_log_radii, total_variance, include_norm=True
        )

    # then integrate this. I restrict the range to ensure convergence. But this is from
    # 10^-5 to 10^5 pc, it will have all the likelihood
    r_term = integrate.quad(integrand_radius, -5, 5)[0]

    # then the final selection probability is the product of these two
    return v_term * r_term


# ======================================================================================
#
# likelihood functions
#
# ======================================================================================
# define the functions to minimize
def log_likelihood(
    params, log_mass, log_mass_err, log_r_eff, log_r_eff_err, log_age, log_age_err
):
    """
    Get the log likelihood for a given model

    :param params: All the free parameters of the model. The first 3 parameters are
                   beta (slope), log(r_4) (normalization), and sigma (intrinsic
                   scatter), followed by the values for underlying masses and ages.
    :param log_mass: Observed log masses.
    :param log_mass_err:  Errors on the observed log mass
    :param log_r_eff: Observed log radii
    :param log_r_eff_err: Errors on the observed log radii
    :param log_age: Observed log age
    :param log_age_err: Errors on the observed log age
    :return:
    """
    # parse the parameters
    beta = params[0]
    log_r_4 = params[1]
    sigma = params[2]
    # then split the ages from the masses
    split_idx = 3 + len(log_mass)
    intrinsic_log_mass = params[3:split_idx]
    intrinsic_log_age = params[split_idx:]
    assert len(intrinsic_log_mass) == len(intrinsic_log_age) == len(log_mass)

    # put priors first, since these short circuit the evaluation
    if (
        abs(beta) > 1
        or sigma < 0
        or sigma > 1
        or abs(log_r_4) > 2
        or np.any(intrinsic_log_mass > max_log_mass)
        or np.any(intrinsic_log_mass < min_log_mass)
        or np.any(intrinsic_log_age > max_log_age)
        or np.any(intrinsic_log_age < min_log_age)
    ):
        return np.inf

    # start by getting the likelihoods of the intrinsic masses and radii. The error is
    # not a free parameter, so we don't need to include the normalization
    log_likelihood = 0
    log_likelihood += np.sum(
        log_gaussian(
            intrinsic_log_mass, log_mass, log_mass_err ** 2, include_norm=False
        )
    )
    log_likelihood += np.sum(
        log_gaussian(intrinsic_log_age, log_age, log_age_err ** 2, include_norm=False)
    )

    # then add the probability of the observed radius from the true mass. In this
    # equation, I'm analytically marginalizing over the true radius. This produces a
    # Gaussian where we compare observed to predicted intrinsic, with the variances
    # added.
    expected_log_radii = mass_size_relation_mean_log(intrinsic_log_mass, beta, log_r_4)
    total_variance = log_r_eff_err ** 2 + sigma ** 2
    # note that here we do need to return the correct normalization, as the scatter is
    # a term of interest
    log_likelihood += np.sum(
        log_gaussian(expected_log_radii, log_r_eff, total_variance, include_norm=True)
    )

    # then normalize by the selection function. In the (not log) likelihood it enters
    # as division, so we subtract the log value. We have to do this separately for
    # each cluster
    selection_likelihoods = [
        selection_probability(
            intrinsic_log_mass[i], intrinsic_log_age[i], beta, log_r_4, sigma
        )
        for i in range(len(log_r_eff))
    ]
    log_likelihood -= np.sum(np.log(np.maximum(0.01, selection_likelihoods)))

    return log_likelihood


def is_converged(sampler, previous_stds):
    # Simple convergence check based on parameters of interest. We iterate until the
    # standard deviation is within some percent of the previous version.
    std_tol = 0.01
    # if we haven't started yet, we haven't converged
    if sampler.iteration == 0:
        return False, previous_stds  # should be initialized to infinity
    # always use a burn-in of 10% of the chain. Not an amazing estimate, but
    # should be good enough. I know the rough values of the parameters, so the
    # burn-in isn't very long anyway
    burn_in = int(0.1 * sampler.iteration)
    samples = sampler.get_chain(flat=True, discard=burn_in, thin=1)  # no thinning here
    stds = np.array([np.median(samples[:, idx]) for idx in range(3)])
    std_diffs = np.abs((stds - previous_stds) / stds)
    converged = np.all(std_diffs < std_tol)
    if converged:
        print(f"converged after {sampler.iteration} iterations!")
    else:
        print(
            f"{sampler.iteration} iterations, "
            f"max sigma change = {100 * np.max(std_diffs):.1f}%"
        )
    return converged, stds


def fit_mass_size_relation(
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    age,
    age_err_lo,
    age_err_hi,
    plots_dir=None,
    plots_prefix="",
):
    log_mass, log_mass_err_lo, log_mass_err_hi = mru.transform_to_log(
        mass, mass_err_lo, mass_err_hi
    )
    log_r_eff, log_r_eff_err_lo, log_r_eff_err_hi = mru.transform_to_log(
        r_eff, r_eff_err_lo, r_eff_err_hi
    )
    log_age, log_age_err_lo, log_age_err_hi = mru.transform_to_log(
        age, age_err_lo, age_err_hi
    )
    # then symmetrixe the errors. I'll start with a simple mean.
    log_mass_err = np.mean([log_mass_err_lo, log_mass_err_hi], axis=0)
    log_r_eff_err = np.mean([log_r_eff_err_lo, log_r_eff_err_hi], axis=0)
    log_age_err = np.mean([log_age_err_lo, log_age_err_hi], axis=0)
    # validate the data passed int
    n_clusters = len(mass)
    assert len(log_mass_err) == len(log_r_eff_err) == len(log_age_err) == n_clusters
    assert len(log_mass) == len(log_r_eff) == len(log_age) == n_clusters

    # Then set up the MCMC.
    # our dimensions for fitting include slope, intercept, scatter, plus mass and age
    # for each cluster
    n_dim = 3 + 2 * n_clusters
    n_walkers = 2 * n_dim + 1  # need at least 2x the dimensions
    args = [log_mass, log_mass_err, log_r_eff, log_r_eff_err, log_age, log_age_err]
    backend = emcee.backends.HDFBackend(f"mcmc_chain_{n_dim}dim.h5")
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_likelihood, args=args, backend=backend
    )

    # make the starting points. If we have already run this sampler and are loading it
    # back, start from the last point
    if sampler.iteration > 0:
        state = sampler.get_last_sample()
    else:  # just starting out, start from reasonable estimates
        # create the array, which has dimensions of walkers, then dimensions
        state = np.zeros((n_walkers, n_dim))
        # then set the relevant portions of it.
        # first slope, intercept, and scatter
        state[:, 0] = np.random.uniform(0, 0.5, n_walkers)
        state[:, 1] = np.random.uniform(0, 1, n_walkers)
        state[:, 2] = np.random.uniform(0.01, 0.5, n_walkers)
        # masses and ages will be perturbed within the errors
        for idx in range(n_clusters):
            masses = log_mass[idx] + np.random.normal(0, log_mass_err[idx], n_walkers)
            ages = log_age[idx] + np.random.normal(0, log_age_err[idx], n_walkers)
            state[:, 3 + idx] = masses
            state[:, 3 + idx + n_clusters] = ages
        # double check that we added everything to the array
        assert not 0 in np.array(state)

    # then run until we're converged!
    stds = np.inf * np.ones(3)  # uninitialized value, will set later
    converged, stds = is_converged(sampler, stds)
    while not converged:
        state = sampler.run_mcmc(state, 1000, progress=True)
        converged, stds = is_converged(sampler, stds)

    # First make a plot of the chains if desired:
    if plots_dir is not None:
        samples = sampler.get_chain(flat=False, discard=0, thin=1)
        plot_chains(samples, plots_dir, plots_prefix)

    # then postprocess this to get the mean values.
    # we throw away the beginning as burn-in, and also thin it.
    # here we actually use the autocorrelation time
    tau = sampler.get_autocorr_time(discard=int(0.1 * sampler.iteration), tol=0)
    n_burn_in = max(int(2 * np.max(tau)), 0.1 * sampler.iteration)
    n_thin = int(np.max(tau))
    samples = sampler.get_chain(flat=True, discard=n_burn_in, thin=n_thin)
    best_fit_params = [np.median(samples[:, idx]) for idx in range(3)]

    return best_fit_params, samples


# ======================================================================================
#
# Debug plots for MCMC
#
# ======================================================================================
def mcmc_plots(
    samples,
    mass,
    mass_err_lo,
    mass_err_hi,
    age,
    age_err_lo,
    age_err_hi,
    ids,
    galaxies,
    plots_dir,
    plots_prefix,
    plot_mass_posteriors=True,
):
    """
    Parent function for the debug MCMC plots - call this externally

    :param samples: The full posterior sample list out of emcee
    :param mass: The observed masses
    :param mass_err_lo: The observed mass lower limits
    :param mass_err_hi: The observed mass upper limits
    :param age: The observed ages
    :param age_err_lo: The observed age lower limits
    :param age_err_hi: The observed age upper limits
    :param ids: The cluster IDs corresponding to the above
    :param galaxies: The galaxy each cluster belongs to
    :param plots_dir: Directory to save these plots to - can be None to not save
    :param plots_prefix: Prefix to the plot savename - will be common to all plots
    :param plot_mass_posteriors: whether or not to make these plots
    :return: None
    """
    # plot the posteriors for the parameters
    plot_params(samples, plots_dir, plots_prefix)

    if plot_mass_posteriors:
        split_idx = 3 + len(mass)
        mass_samples = 10 ** samples[:, 3:split_idx]
        age_samples = 10 ** samples[:, split_idx:]

        for galaxy in np.unique(galaxies):
            gal_mask = np.where(galaxies == galaxy)[0]

            plot_cluster_samples(
                mass_samples[:, gal_mask],
                mass[gal_mask],
                mass_err_lo[gal_mask],
                mass_err_hi[gal_mask],
                ids[gal_mask],
                galaxy,
                "mass",
                plots_dir,
                plots_prefix,
            )
            plot_cluster_samples(
                age_samples[:, gal_mask],
                age[gal_mask],
                age_err_lo[gal_mask],
                age_err_hi[gal_mask],
                ids[gal_mask],
                galaxy,
                "age",
                plots_dir,
                plots_prefix,
            )


def plot_params(samples, plots_dir, plots_prefix):
    # plot the posterior for the fit parameters
    fig = corner.corner(
        samples[:, :3],
        labels=["$\\beta$", "log($r_4$)", "$\sigma$"],
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 14},
        label_kwargs={"fontsize": 14},
    )

    if not plots_dir is None:
        fig.savefig(plots_dir / f"{plots_prefix}_param_posterior.png", dpi=100)


def plot_chains(samples, plots_dir, plots_prefix):
    fig, axs = bpl.subplots(nrows=7, sharex=True, figsize=[10, 10])
    # plot the 3 main parameters plus 2 mass and age chains, randomly selected
    n_iterations, n_walkers, n_params = samples.shape
    n_clusters = 0.5 * (n_params - 3)
    assert np.isclose(int(n_clusters), n_clusters)
    n_clusters = int(n_clusters)
    param_idxs = np.concatenate(
        [
            [0, 1, 2],
            np.random.randint(3, 3 + n_clusters, 2),
            np.random.randint(3 + n_clusters, n_params, 2),
        ]
    )

    names = [
        "$\\beta$",
        "log($r_4$)",
        "$\sigma$",
        "log($m_{" + f"{param_idxs[-4]}" + "}$)",
        "log($m_{" + f"{param_idxs[-3]}" + "}$)",
        "log($t_{" + f"{param_idxs[-4]}" + "}$)",
        "log($t_{" + f"{param_idxs[-3]}" + "}$)",
    ]

    # x values are simply the position
    xs = np.arange(1, n_iterations + 1, 1)

    for ax, param_idx, name in zip(axs, param_idxs, names):
        for chain_idx in range(n_walkers):
            ax.plot(xs, samples[:, chain_idx, param_idx], lw=0.1, c=bpl.almost_black)
        ax.add_labels(y_label=name)
        ax.set_limits(x_min=1, x_max=n_iterations)

    axs[-1].add_labels(x_label="Iteration")

    if not plots_dir is None:
        fig.savefig(plots_dir / f"{plots_prefix}_chains.png", dpi=100)


def plot_cluster_samples(
    samples, x, x_err_lo, x_err_hi, ids, galaxy, value, plots_dir, plots_prefix
):
    # Plot the posteriors for the mass/radius of a given set of clusters

    # first plot the measured values. Here each cluster will have it's own row with a
    # dummy y value
    fig, ax = bpl.subplots(figsize=[7, 2 + 0.4 * len(x)])
    ax.make_ax_dark()
    dummy_y = np.arange(0, len(x))
    offset = 0.15
    ax.errorbar(
        x,
        dummy_y + offset,
        xerr=[x_err_lo, x_err_hi],
        c=bpl.color_cycle[0],
        label="Measurement Error",
    )
    # also show the symmetrized error. The getting of this error is copied
    # from the MCMC function.
    log_x, log_x_err_lo, log_x_err_hi = mru.transform_to_log(x, x_err_lo, x_err_hi)
    log_x_err_symm = np.mean([log_x_err_lo, log_x_err_hi], axis=0)
    x_err_lo_logsymm = x - 10 ** (log_x - log_x_err_symm)
    x_err_hi_logsymm = 10 ** (log_x + log_x_err_symm) - x
    ax.errorbar(
        x,
        dummy_y,
        xerr=[x_err_lo_logsymm, x_err_hi_logsymm],
        c=bpl.color_cycle[2],
        label="Symmetrized Error",
    )

    # Then go through and plot the posteriors for each cluster
    for idx in range(len(dummy_y)):
        y = dummy_y[idx] - offset
        this_samples = samples[:, idx]

        # use the percentiles of the posterior as the error range
        lo, median, hi = np.percentile(this_samples, [16, 50, 84])
        err_lo = median - lo
        err_hi = hi - median

        # only label the first datapoint
        if idx == 0:
            label = "Posterior"
        else:
            label = None

        # not sure why the error is so complicated here
        ax.errorbar(
            [median],
            [y],
            xerr=((err_lo,), (err_hi,)),
            c=bpl.color_cycle[3],
            label=label,
        )

    # then format the axes. set the y labels to be the cluster IDs
    ax.set_yticks(dummy_y)
    ax.set_yticklabels(ids)
    if value == "mass":
        ax.set_limits(100, 1e6, min(dummy_y) - 1, max(dummy_y) + 1)
        ax.add_labels("Mass [$M_\odot$]", "ID", galaxy.replace("ngc", "NGC "))
    elif value == "radius":
        ax.set_limits(0.1, 30, min(dummy_y) - 1, max(dummy_y) + 1)
        ax.add_labels("Radius [pc]", "ID", galaxy.replace("ngc", "NGC "))
    elif value == "age":
        ax.set_limits(1e6, 1e11, min(dummy_y) - 1, max(dummy_y) + 1)
        ax.add_labels("Age [yr]", "ID", galaxy.replace("ngc", "NGC "))
    ax.set_xscale("log")
    ax.legend()

    if not plots_dir is None:
        fig.savefig(plots_dir / f"{plots_prefix}_{galaxy}_{value}_posterior.png")
