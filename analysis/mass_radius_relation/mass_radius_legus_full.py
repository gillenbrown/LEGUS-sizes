"""
mass_radius_legus_full.py
- Fit the mass-size relation for all LEGUS clusters
"""
import sys
from pathlib import Path

import numpy as np
import betterplotlib as bpl

import mass_radius_utils as mru
import mass_radius_utils_mle_fitting as mru_mle
import mass_radius_utils_mcmc_fitting as mru_mcmc
import mass_radius_utils_plotting as mru_p

bpl.set_style()

# load the parameters the user passed in
plot_name = Path(sys.argv[1])
output_name = Path(sys.argv[2])
fit_out_file = open(output_name, "w")
mcmc_plot_dir = Path(sys.argv[3])
big_catalog = mru.make_big_table(sys.argv[4:])

# start parsing my catalogs
# No clusters to filter out here
# mask = np.full(len(big_catalog), True)
mask = big_catalog["ID"] < 10  # restrict sample size for testing
mass, mass_err_lo, mass_err_hi = mru.get_my_masses(big_catalog, mask)
r_eff, r_eff_err_lo, r_eff_err_hi = mru.get_my_radii(big_catalog, mask)
ids = big_catalog["ID"][mask]
galaxies = big_catalog["galaxy"][mask]

# restrict to mass range. Can't do this at the beginning since we want to
# plot everything
fit_mask = mass < 1e5
print(np.sum(fit_mask))

# Then actually make the fit and plot it. Do this for both MLE and MCMC
fit_mle, fit_history_mle = mru_mle.fit_mass_size_relation(
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    fit_mass_upper_limit=1e5,
)

fit_mcmc, fit_history_mcmc = mru_mcmc.fit_mass_size_relation(
    mass[fit_mask],
    mass_err_lo[fit_mask],
    mass_err_hi[fit_mask],
    r_eff[fit_mask],
    r_eff_err_lo[fit_mask],
    r_eff_err_hi[fit_mask],
)
# make the debug plots for the MCMC chain
mru_mcmc.mcmc_plots(
    fit_history_mcmc,
    mass[fit_mask],
    mass_err_lo[fit_mask],
    mass_err_hi[fit_mask],
    ids[fit_mask],
    galaxies[fit_mask],
    mcmc_plot_dir,
    "legus_full",
    True,
)


# then plot the dataset
fig, ax = bpl.subplots()
mru_p.plot_mass_size_dataset_scatter(
    ax,
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    bpl.color_cycle[0],
)
# mru_p.add_percentile_lines(ax, mass, r_eff)
mru_p.plot_best_fit_line(ax, fit_mle, 1e2, 1e5, color=bpl.color_cycle[1], label="MLE")
mru_p.plot_best_fit_line(ax, fit_mcmc, 1e2, 1e5, color=bpl.color_cycle[2], label="MCMC")
mru_p.format_mass_size_plot(ax)
fig.savefig(plot_name)

mru.write_fit_results(
    fit_out_file, "Full LEGUS Sample - MLE", len(r_eff), fit_mle, fit_history_mle
)
mru.write_fit_results(
    fit_out_file,
    "Full LEGUS Sample - MCMC",
    len(r_eff),
    fit_mcmc,
    fit_history_mcmc[:, :3],
)

# finalize output file
fit_out_file.close()
