"""
mass_radius_legus_ssfrsplit.py
- Fit the mass-size relation for all LEGUS clusters, splitting by sSFR
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
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

# Load galaxy sSFR data
calzetti_path = plot_name.parent.parent / "analysis" / "calzetti_etal_15_table_1.txt"
galaxy_table = table.Table.read(
    calzetti_path, format="ascii.commented_header", header_start=3
)
# Add the data to my cluster tables
for col in ["galaxy_SFR", "galaxy_m_star", "galaxy_t_type", "galaxy_sSFR"]:
    big_catalog[col] = np.nan
for row in big_catalog:  # need to account for NGC 5194/NGC5195 in same catalog
    # throw away east-west-north-south field splits
    this_gal = row["galaxy"].split("-")[0].upper()
    for row_g in galaxy_table:
        if row_g["name"] == this_gal:
            row["galaxy_SFR"] = row_g["sfr_uv_msun_per_year"]
            row["galaxy_m_star"] = row_g["m_star"]
            row["galaxy_t_type"] = row_g["morphology_t_type"]
            row["galaxy_sSFR"] = row_g["sfr_uv_msun_per_year"] / row_g["m_star"]
            break

# filter out clusters 1 Gyr or older
mask = big_catalog["age_yr"] < 1e9
mass, mass_err_lo, mass_err_hi = mru.get_my_masses(big_catalog, mask)
r_eff, r_eff_err_lo, r_eff_err_hi = mru.get_my_radii(big_catalog, mask)
age, _, _ = mru.get_my_ages(big_catalog, mask)
ssfr = big_catalog["galaxy_sSFR"][mask]

# Then actually make the fit and plot it
cut_ssfr = 3e-10
mask_hi_ssfr = np.logical_and(ssfr >= cut_ssfr, age < 1e9)
mask_lo_ssfr = np.logical_and(ssfr < cut_ssfr, age < 1e9)

fig, ax = bpl.subplots()
for mask_ssfr, name, color in zip(
    [mask_lo_ssfr, mask_hi_ssfr],
    [
        "sSFR $< 3 \\times 10^{-10} {\\rm yr}^{-1}$",
        "sSFR $\geq 3 \\times 10^{-10} {\\rm yr}^{-1}$",
    ],
    [bpl.color_cycle[3], bpl.color_cycle[0]],
):
    fit, fit_history = mru_mle.fit_mass_size_relation(
        mass[mask_ssfr],
        mass_err_lo[mask_ssfr],
        mass_err_hi[mask_ssfr],
        r_eff[mask_ssfr],
        r_eff_err_lo[mask_ssfr],
        r_eff_err_hi[mask_ssfr],
        fit_mass_upper_limit=1e5,
    )

    mru_p.plot_mass_size_dataset_contour(
        ax,
        mass[mask_ssfr],
        r_eff[mask_ssfr],
        color,
    )
    # add_percentile_lines(ax, mass_legus[mask], r_eff_legus[mask], color=color)
    mru_p.plot_best_fit_line(
        ax, fit, 1, 1e5, color, fill=False, label=f"{name}, N={np.sum(mask_ssfr)}"
    )
    mru.write_fit_results(fit_out_file, name, np.sum(mask_ssfr), fit, fit_history)
mru_p.format_mass_size_plot(ax)
fig.savefig(plot_name)

# finalize output file
fit_out_file.close()
