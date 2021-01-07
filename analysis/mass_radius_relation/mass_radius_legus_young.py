"""
mass_radius_legus_young.py
- Fit the mass-size relation for LEGUS clusters younger than 1 Gyr
"""
import sys
from pathlib import Path

import betterplotlib as bpl

import mass_radius_utils as mru
import mass_radius_utils_mle_fitting as mru_mle

bpl.set_style()

# load the parameters the user passed in
output_name = Path(sys.argv[1])
fit_out_file = open(output_name, "w")
big_catalog = mru.make_big_table(sys.argv[2:])

# Filter out clusters older than 1 Gyr
mask = big_catalog["age_yr"] < 1e9
mass, mass_err_lo, mass_err_hi = mru.get_my_masses(big_catalog, mask)
r_eff, r_eff_err_lo, r_eff_err_hi = mru.get_my_radii(big_catalog, mask)

# Then actually make the fit, don't plot it
fit, fit_history = mru_mle.fit_mass_size_relation(
    mass,
    mass_err_lo,
    mass_err_hi,
    r_eff,
    r_eff_err_lo,
    r_eff_err_hi,
    fit_mass_upper_limit=1e5,
)
mru.write_fit_results(fit_out_file, "1 Myr - 1 Gyr", fit, fit_history, mass)

# finalize output file
fit_out_file.close()
