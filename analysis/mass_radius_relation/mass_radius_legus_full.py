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
import mass_radius_utils_plotting as mru_p

bpl.set_style()

# load the parameters the user passed in
plot_name = Path(sys.argv[1])
output_name = Path(sys.argv[2])
fit_out_file = open(output_name, "w")
big_catalog = mru.make_big_table(sys.argv[3:])

# start parsing my catalogs
# No clusters to filter out here
mask = np.full(len(big_catalog), True)
mass, mass_err_lo, mass_err_hi = mru.get_my_masses(big_catalog, mask)
r_eff, r_eff_err_lo, r_eff_err_hi = mru.get_my_radii(big_catalog, mask)

# restrict to mass range. Can't do this at the beginning since we want to
# plot everything
fit_mask = mass < 1e5

# Then actually make the fit and plot it. Do this for both MLE and MCMC
fit, fit_history = mru_mle.fit_mass_size_relation(
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    fit_mass_upper_limit=1e5,
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
mru_p.plot_best_fit_line(ax, fit, 1e2, 1e5, color=bpl.color_cycle[1])
mru_p.format_mass_size_plot(ax)
fig.savefig(plot_name)

mru.write_fit_results(fit_out_file, "Full LEGUS Sample", len(r_eff), fit, fit_history)

# finalize output file
fit_out_file.close()
