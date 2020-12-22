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
age, age_err_lo, age_err_hi = mru.get_my_ages(big_catalog, mask)
ids = big_catalog["ID"][mask]
galaxies = big_catalog["galaxy"][mask]

# restrict to mass range. Can't do this at the beginning since we want to
# plot everything
fit_mask = mass < 1e5
print(np.sum(fit_mask))

# Then actually make the fit and plot it. Do this for both MLE and MCMC
fit_mle_orthogonal, fit_history_mle_orthogonal = mru_mle.fit_mass_size_relation(
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    fit_mass_upper_limit=1e5,
    fit_style="orthogonal",
)
fit_mle_vertical, fit_history_mle_vertical = mru_mle.fit_mass_size_relation(
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    fit_mass_upper_limit=1e5,
    fit_style="vertical",
)

fit_mcmc_no_select, fit_history_mcmc_no_select = mru_mcmc.fit_mass_size_relation(
    mass[fit_mask],
    mass_err_lo[fit_mask],
    mass_err_hi[fit_mask],
    r_eff[fit_mask],
    r_eff_err_lo[fit_mask],
    r_eff_err_hi[fit_mask],
    age[fit_mask],
    age_err_lo[fit_mask],
    age_err_hi[fit_mask],
    mcmc_plot_dir,
    "legus_full_no_selection",
    v_band_cut=None,
)

fit_mcmc_all_select, fit_history_mcmc_all_select = mru_mcmc.fit_mass_size_relation(
    mass[fit_mask],
    mass_err_lo[fit_mask],
    mass_err_hi[fit_mask],
    r_eff[fit_mask],
    r_eff_err_lo[fit_mask],
    r_eff_err_hi[fit_mask],
    age[fit_mask],
    age_err_lo[fit_mask],
    age_err_hi[fit_mask],
    mcmc_plot_dir,
    "legus_full_all_selection",
    v_band_cut=100,  # dummy value to show that we select everything
    # should be identical to fitting without including selection
)

fit_mcmc_real_select, fit_history_mcmc_real_select = mru_mcmc.fit_mass_size_relation(
    mass[fit_mask],
    mass_err_lo[fit_mask],
    mass_err_hi[fit_mask],
    r_eff[fit_mask],
    r_eff_err_lo[fit_mask],
    r_eff_err_hi[fit_mask],
    age[fit_mask],
    age_err_lo[fit_mask],
    age_err_hi[fit_mask],
    mcmc_plot_dir,
    "legus_full_real_selection",
    v_band_cut=-6,
)

# make the debug plots for the MCMC chain
mru_mcmc.mcmc_plots(
    fit_history_mcmc_real_select,
    mass[fit_mask],
    mass_err_lo[fit_mask],
    mass_err_hi[fit_mask],
    age[fit_mask],
    age_err_lo[fit_mask],
    age_err_hi[fit_mask],
    ids[fit_mask],
    galaxies[fit_mask],
    mcmc_plot_dir,
    "legus_full_real_selection",
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
for fit, history, name, color in zip(
    [
        fit_mle_orthogonal,
        fit_mle_vertical,
        fit_mcmc_no_select,
        fit_mcmc_all_select,
        fit_mcmc_real_select,
    ],
    [
        fit_history_mle_orthogonal,
        fit_history_mle_vertical,
        fit_history_mcmc_no_select.T,
        fit_history_mcmc_all_select.T,
        fit_history_mcmc_real_select.T,
    ],
    [
        "Orthogonal",
        "Vertical",
        "MCMC - No Selection",
        "MCMC - V$_{cut}$=100",
        "MCMC - V$_{cut}$=-6",
    ],
    bpl.color_cycle[:5],
):
    label = f"{name} - "
    for param_idx, param_name in zip([0, 1, 2], ["$\\beta$", "log$r_4$", "$\sigma$"]):
        lo, hi = np.percentile(history[param_idx], [16, 84])
        med = fit[param_idx]
        label += f"{param_name} = {med:.2f}"
        label += "$^{" + f"{hi - med:.2f}" + "}"
        label += "_{" + f"{med - lo:.2f}" + "}$, "
    mru_p.plot_best_fit_line(ax, fit, 1e2, 1e5, color=color, label=label, fill=False)

# mru_p.plot_best_fit_line(ax, fit_mle, 1e2, 1e5, color=bpl.color_cycle[1], label="MLE")
# mru_p.plot_best_fit_line(ax, fit_mcmc, 1e2, 1e5, color=bpl.color_cycle[2], label="MCMC")
mru_p.format_mass_size_plot(ax, legend_fontsize=12)
fig.savefig(plot_name)

# mru.write_fit_results(
#     fit_out_file, "Full LEGUS Sample - MLE", len(r_eff), fit_mle, fit_history_mle
# )
# mru.write_fit_results(
#     fit_out_file,
#     "Full LEGUS Sample - MCMC",
#     len(r_eff),
#     fit_mcmc,
#     fit_history_mcmc[:, :3].T,
# )

# finalize output file
fit_out_file.close()
