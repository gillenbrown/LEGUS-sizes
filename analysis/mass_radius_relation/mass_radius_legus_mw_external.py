"""
mass_radius_legus_mw_external.py
- Fit the mass-size relation for all LEGUS clusters plus clusters from the MW and M31
"""
import sys
from pathlib import Path

import numpy as np
import betterplotlib as bpl

import mass_radius_utils as mru
import mass_radius_utils_mle_fitting as mru_mle
import mass_radius_utils_plotting as mru_p
import mass_radius_utils_external_data as mru_d

bpl.set_style()

# load the parameters the user passed in
plot_name = Path(sys.argv[1])
output_name = Path(sys.argv[2])
fit_out_file = open(output_name, "w")
big_catalog = mru.make_big_table(sys.argv[3:])

# Filter out clusters older than 1 Gyr
mask = big_catalog["age_yr"] < 1e9
mass, mass_err_lo, mass_err_hi = mru.get_my_masses(big_catalog, mask)
r_eff, r_eff_err_lo, r_eff_err_hi = mru.get_my_radii(big_catalog, mask)

# Then I need to combine all the datasets together. To do this, I iterate through the
# functions that are used to get the data, then concatenate everything together. This
# also lets me plot the datasets one by one. But to do this I need a dummy function to
# handle the legus data
def dummy_legus():
    return mass, mass_err_lo, mass_err_hi, r_eff, r_eff_err_lo, r_eff_err_hi


mass_total = np.array([])
mass_err_lo_total = np.array([])
mass_err_hi_total = np.array([])
r_eff_total = np.array([])
r_eff_err_lo_total = np.array([])
r_eff_err_hi_total = np.array([])
fig, ax = bpl.subplots()
for func, label, color in zip(
    [
        dummy_legus,
        mru_d.get_mw_open_clusters,
        mru_d.get_m31_open_clusters,
        mru_d.get_m83_clusters,
        mru_d.get_mw_ymc_krumholz_19_clusters,
        mru_d.get_m82_sscs,
        mru_d.get_ngc253_sscs,
    ],
    ["LEGUS", "MW OCs", "M31 OCs", "M83", "MW YMCs", "M82 SSCs", "NGC 253 SSCs"],
    [
        bpl.color_cycle[0],
        bpl.color_cycle[2],
        bpl.color_cycle[3],
        bpl.color_cycle[4],
        bpl.color_cycle[5],
        bpl.color_cycle[6],
        bpl.color_cycle[7],
    ],
):
    # get the data from this function
    m, m_el, m_eh, r, r_el, r_eh = func()
    print(f"Including {len(m)} clusters from {label}")
    # and plot it
    mru_p.plot_mass_size_dataset_scatter(ax, m, m_el, m_eh, r, r_el, r_eh, color, label)
    # and add the data to the total
    mass_total = np.concatenate([mass_total, m])
    mass_err_lo_total = np.concatenate([mass_err_lo_total, m_el])
    mass_err_hi_total = np.concatenate([mass_err_hi_total, m_eh])
    r_eff_total = np.concatenate([r_eff_total, r])
    r_eff_err_lo_total = np.concatenate([r_eff_err_lo_total, r_el])
    r_eff_err_hi_total = np.concatenate([r_eff_err_hi_total, r_eh])


# Then we can do the fit
fit, fit_history = mru_mle.fit_mass_size_relation(
    mass_total,
    mass_err_lo_total,
    mass_err_hi_total,
    r_eff_total,
    r_eff_err_lo_total,
    r_eff_err_hi_total,
    fit_mass_upper_limit=1e5,
)

mru_p.add_percentile_lines(ax, mass_total, r_eff_total)
mru_p.plot_best_fit_line(ax, fit, 1, 1e5, label_intrinsic_scatter=False)
mru_p.format_mass_size_plot(ax, xmin=1, xmax=1e7)
fig.savefig(plot_name)
mru.write_fit_results(
    fit_out_file, "LEGUS + MW + External Galaxies", fit, fit_history, mass_total
)

# finalize output file
fit_out_file.close()
