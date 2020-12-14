import numpy as np
import emcee

import mass_radius_utils as mru
import corner
import betterplotlib as bpl

bpl.set_style()


# define the functions to minimize
def log_likelihood(params, log_mass, log_mass_err, log_r_eff, log_r_eff_err):
    slope = params[0]
    y_at_pivot = params[1]
    scatter = params[2]
    # first convert the pivot point value into the intercept
    pivot_point_x = 4
    # so params[1] is really y(pivot_point_x) = m (pivot_point_x) + intercept
    intercept = y_at_pivot - slope * pivot_point_x

    # then get the intrinsic masses
    intrinsic_log_mass = params[3:]
    assert len(intrinsic_log_mass) == len(log_mass)

    # start by getting the likelihoods of the intrinsic masses
    log_likelihood = 0
    log_likelihood += -0.5 * np.sum(
        ((intrinsic_log_mass - log_mass) / log_mass_err) ** 2
    )

    # then add the probability of the observed radius from the true mass. In this
    # equation, I'm analytically marginalizing over the true radius. This produces a
    # Gaussian where we compare observed to predicted intrinsic, with the variances
    # added.
    expected_true_log_radii = intercept + intrinsic_log_mass * slope
    total_variance = log_r_eff_err ** 2 + scatter ** 2
    log_likelihood += -0.5 * np.sum(
        ((expected_true_log_radii - log_r_eff) ** 2) / total_variance
    )

    # then penalize large intrinsic scatter. This term really comes from the definition
    # of a Gaussian likelihood. This term is always out front of a Gaussian, but
    # normally it's just a constant. When we include intrinsic scatter it now
    # affects the likelihood.
    log_likelihood += np.sum(-0.5 * np.log(total_variance))

    #     # priors
    if abs(slope) > 1 or scatter < 0:
        log_likelihood -= np.inf

    return log_likelihood


def is_converged(sampler, previous_tau):
    # This convergence code inspired by:
    # https://emcee.readthedocs.io/en/stable/tutorials/monitor/
    autocorr_multiples = 100
    autocorr_change_tol = 0.1
    # check for samplers that aren't initialized yet
    if sampler.iteration < 1:
        return False, previous_tau  # will be set to infinity at the beginning anyway
    elif sampler.iteration > 2000:
        return True, 100  # dummy values
    # Compute the autocorrelation time so far. Set the tolerance for trusting the
    # autocorrelation time. This is the value that our chain must be X times longer
    # than the calculated value to trust that value.
    tol = 10
    # set the burn in
    if not np.any(np.isinf(previous_tau)):
        burn_in = int(2 * np.max(previous_tau))  # 2 autocorrelation times
    else:
        burn_in = int(0.1 * sampler.iteration)  # 10% of the chain
    # This will raise an error if the chain isn't long enough to trust the
    # autocorrelation time
    try:
        tau = sampler.get_autocorr_time(tol=tol, discard=burn_in)
    except emcee.autocorr.AutocorrError:  # too short to trust
        est_tau = sampler.get_autocorr_time(tol=0, discard=burn_in)
        print(
            f"{sampler.iteration} iterations isn't enough to trust autocorr time,"
            f" estimated to need {np.max(est_tau) * tol + burn_in:.0f}"
        )
        return False, est_tau
    # if we're here the autocorrelation time is reliable
    # check convergence.
    converged = np.all(np.abs(previous_tau - tau) / tau < autocorr_change_tol)
    converged &= np.max(tau) * autocorr_multiples < sampler.iteration

    if converged:
        print(f"converged after {sampler.iteration} iterations!")
    else:
        n_needed = autocorr_multiples * np.ceil(np.max(tau)) + burn_in
        print(f"{sampler.iteration} iterations, estimated to need {n_needed:.0f}")
    return converged, tau


def fit_mass_size_relation(
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
):
    log_mass, log_mass_err_lo, log_mass_err_hi = mru.transform_to_log(
        mass, mass_err_lo, mass_err_hi
    )
    log_r_eff, log_r_eff_err_lo, log_r_eff_err_hi = mru.transform_to_log(
        r_eff, r_eff_err_lo, r_eff_err_hi
    )
    # then symmetrixe the errors. I'll start with a simple mean.
    log_mass_err = np.mean([log_mass_err_lo, log_mass_err_hi], axis=0)
    log_r_eff_err = np.mean([log_r_eff_err_lo, log_r_eff_err_hi], axis=0)
    assert len(log_mass_err) == len(log_r_eff_err) == len(mass) == len(r_eff)

    # Then set up the MCMC.
    # our dimensions for fitting include slope, intercept, scatter, plus mass
    # for each cluster
    n_dim = 3 + len(log_mass)
    n_walkers = 2 * n_dim + 1  # need at least 2x the dimensions
    args = [log_mass, log_mass_err, log_r_eff, log_r_eff_err]
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
        # masses will be perturbed within the errors
        for idx in range(len(log_mass)):
            masses = log_mass[idx] + np.random.normal(0, log_mass_err[idx], n_walkers)
            state[:, 3 + idx] = masses
        # double check that we added everything to the array
        assert not 0 in np.array(state)

    # then run until we're converged!
    tau = np.inf * np.ones(n_dim)  # uninitialized value, will set later
    converged, tau = is_converged(sampler, tau)
    while not converged:
        state = sampler.run_mcmc(state, 1000, progress=True)
        converged, tau = is_converged(sampler, tau)

    # then postprocess this to get the mean values.
    # we throw away the beginning as burn-in, and also thin it
    n_burn_in = int(2 * np.max(tau))
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
        mass_samples = 10 ** samples[:, 3:]

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


def plot_params(samples, plots_dir, plots_prefix):
    # plot the posterior for the fit parameters
    fig = corner.corner(
        samples[:, :3],
        labels=["Slope", "log($R_{eff}$) at $10^4 M_\odot$", "Intrinsic Scatter"],
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
    )

    if not plots_dir is None:
        fig.savefig(plots_dir / f"{plots_prefix}_param_posterior.png", dpi=100)


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
    ax.set_xscale("log")
    ax.legend()

    if not plots_dir is None:
        fig.savefig(plots_dir / f"{plots_prefix}_{galaxy}_{value}_posterior.png")
