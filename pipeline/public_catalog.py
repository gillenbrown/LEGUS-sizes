"""
public_catalog.py

Format the catalog nicely and make it fit for public consumption.

Takes the following command line arguments:
- Name to save the output catalog as
- All the catalogs for individual galaxies
"""

import sys
from pathlib import Path
from astropy import table
import numpy as np

# Get the input arguments
output_table = Path(sys.argv[1])
# load the individual catalogs, then stack them together in one master catalog
gal_catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[2:]]
catalog = table.vstack(gal_catalogs, join_type="inner")

# ======================================================================================
#
# a bit of into about the sample
#
# ======================================================================================
n_r = np.sum(catalog["reliable_radius"])
n_rm = np.sum(np.logical_and(catalog["reliable_mass"], catalog["reliable_radius"]))
print(f"{len(catalog)} total clusters")
print(f"{n_r} clusters have reliable radii")
print(f"{n_rm} of those have reliable mass")

print(f"{len(np.unique(catalog['field']))} different fields")
# for field in np.unique(catalog["field"]):
#     print(f"\t- {field}")
print(f"{len(np.unique(catalog['galaxy']))} different galaxies")
# for gal in np.unique(catalog["galaxy"]):
#     print(f"\t- {gal}")

# and do some validation
for col in catalog.colnames:
    try:
        assert np.sum(np.isnan(catalog[col])) == 0
        assert np.sum(np.isinf(catalog[col])) == 0
        assert np.sum(np.isneginf(catalog[col])) == 0
    except TypeError:  # can't check if strings are nans
        continue
    except AssertionError:
        # density and crossing time have nans where the mass is bad
        assert "density" in col or "crossing_time" in col
        # check that there are no nans where the mass is good
        mask_good_mass = catalog["reliable_mass"]
        assert np.sum(np.isnan(catalog[col][mask_good_mass])) == 0
        assert np.sum(np.isinf(catalog[col][mask_good_mass])) == 0
        assert np.sum(np.isneginf(catalog[col][mask_good_mass])) == 0


# ======================================================================================
#
# Formatting the table
#
# ======================================================================================
# delete a few quantities that I calculated for debugging or testing that are not needed
catalog.remove_columns(
    [
        "estimated_local_background_diff_sigma",
        "fit_rms",
        "x_pix_snapshot_oversampled_best",
        "y_pix_snapshot_oversampled_best",
        "x_pix_snapshot_oversampled_e-",
        "x_pix_snapshot_oversampled_e+",
        "y_pix_snapshot_oversampled_e-",
        "y_pix_snapshot_oversampled_e+",
    ]
)


# set the order for the leftover columns
new_col_order = [
    "field",
    "ID",
    "galaxy",
    "distance_mpc",
    "from_ml",
    "RA",
    "Dec",
    "x_pix_single",
    "y_pix_single",
    "mag_F275W",
    "photoerr_F275W",
    "mag_F336W",
    "photoerr_F336W",
    "mag_F814W",
    "photoerr_F814W",
    "CI",
    "age_yr",
    "age_yr_max",
    "age_yr_min",
    "mass_msun",
    "mass_msun_max",
    "mass_msun_min",
    "E(B-V)",
    "E(B-V)_max",
    "E(B-V)_min",
    "chi_2_F265W",
    "chi_2_F336W",
    "chi_2_F814W",
    "chi_2_reduced",
    "Q_probability",
    "N_filters",
    "x",
    "y",
    "x_fitted_best",
    "x_fitted_e+",
    "x_fitted_e-",
    "y_fitted_best",
    "y_fitted_e+",
    "y_fitted_e-",
    "log_luminosity_best",
    "log_luminosity_e+",
    "log_luminosity_e-",
    "scale_radius_pixels_best",
    "scale_radius_pixels_e+",
    "scale_radius_pixels_e-",
    "axis_ratio_best",
    "axis_ratio_e+",
    "axis_ratio_e-",
    "position_angle_best",
    "position_angle_e+",
    "position_angle_e-",
    "power_law_slope_best",
    "power_law_slope_e+",
    "power_law_slope_e-",
    "local_background_best",
    "local_background_e+",
    "local_background_e-",
    "r_eff_pixels",
    "r_eff_pixels_e+",
    "r_eff_pixels_e-",
    "r_eff_arcsec",
    "r_eff_arcsec_e+",
    "r_eff_arcsec_e-",
    "r_eff_pc",
    "r_eff_pc_e+",
    "r_eff_pc_e-",
    "num_boostrapping_iterations",
    "crossing_time_yr",
    "crossing_time_log_err",
    "3d_density",
    "3d_density_log_err",
    "surface_density",
    "surface_density_log_err",
    "profile_diff_reff",
    "radius_fit_failure",
    "reliable_radius",
    "reliable_mass",
    "x_pix_single_fitted",
    "y_pix_single_fitted",
]

# check that I got all columns listed
assert len(new_col_order) == len(catalog.colnames)
assert sorted(new_col_order) == sorted(catalog.colnames)

# then apply this order
catalog = catalog[new_col_order]

# then write the catalog!
catalog.write(output_table, format="ascii.ecsv")
