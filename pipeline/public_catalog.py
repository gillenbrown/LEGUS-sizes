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

import utils

# Get the input arguments
output_table = Path(sys.argv[1])
catalogs = [table.Table.read(item, format="ascii.ecsv") for item in sys.argv[2:]]

# ======================================================================================
#
# handling class
#
# ======================================================================================
# this is ugly, sorry
for cat in catalogs:
    # manually do the sorting of classes depending on the field.
    # `pipeline/format_catalogs.py` was referenced to get this right.
    assert len(np.unique(cat["field"])) == 1
    field = cat["field"][0]

    # set dummy column to make sure the string is long enough
    cat["morphology_class_source"] = " " * 20

    if field == "ngc5194-ngc5195-mosaic":
        # Here we use the class_mode_human, but then supplement it with the ML
        # classification for ones that weren't classified by humans
        mask_ml = cat["class_mode_human"] == 0
        mask_h = ~mask_ml

        # then fill in appropriately. First use dummy for class, will fill in
        cat["morphology_class"] = -99
        cat["morphology_class"][mask_h] = cat["class_mode_human"][mask_h]
        cat["morphology_class"][mask_ml] = cat["class_ml"][mask_ml]
        cat["morphology_class_source"][mask_h] = "human_mode"
        cat["morphology_class_source"][mask_ml] = "ml"

        # check my work
        mask_ml_check = cat["morphology_class_source"] == "ml"
        assert np.array_equal(np.unique(cat["class_mode_human"][mask_ml_check]), [0])

    # note that NGC4449 is another case where we originally selected ML clusters, but
    # now it only has human selected clusters. We threw out the ML clusters in derived
    # properties. Double check this
    elif field == "ngc4449":
        for c in np.unique(cat["class_whitmore"]):
            assert c in [1, 2]
        cat.rename_column("class_whitmore", "morphology_class")
        cat["morphology_class_source"] = "human_mode"

    elif field == "ngc1566":
        # Here we simply use the hybrid method, as the documentation says it is the one
        # to use
        cat.rename_column("class_hybrid", "morphology_class")
        cat["morphology_class_source"] = "hybrid"

    else:  # normal catalogs
        if "class_mode_human" in cat.colnames:
            class_col = "class_mode_human"
        elif "class_linden_whitmore" in cat.colnames:
            class_col = "class_linden_whitmore"
        elif "class_whitmore" in cat.colnames:
            class_col = "class_whitmore"
        else:
            raise ValueError(f"No class found for {field}")

        cat.rename_column(class_col, "morphology_class")
        cat["morphology_class_source"] = "human_mode"


# then stack them together in one master catalog
catalog = table.vstack(catalogs, join_type="inner")

# validate what I've done with the classes
assert np.array_equal(np.unique(catalog["morphology_class"]), [1, 2])
assert np.array_equal(
    np.unique(catalog["morphology_class_source"]),
    ["human_mode", "hybrid", "ml"],
)

# ======================================================================================
#
# galaxy data
#
# ======================================================================================
# then get the stellar mass, SFR, and galaxy distance
home_dir = Path(__file__).parent.parent
galaxy_table = table.Table.read(
    home_dir / "pipeline" / "calzetti_etal_15_table_1.txt",
    format="ascii.commented_header",
    header_start=3,
)
# read the Calzetti table
gal_mass = dict()
gal_sfr = dict()
for row in galaxy_table:
    name = row["name"].lower()
    gal_mass[name] = row["m_star"]
    gal_sfr[name] = row["sfr_uv_msun_per_year"]

# set dummy quantities
catalog["galaxy_distance_mpc"] = -99.9
catalog["galaxy_distance_mpc_err"] = -99.9
catalog["galaxy_stellar_mass"] = -99.9
catalog["galaxy_sfr"] = -99.9

# then add these quantities for all columns
for row in catalog:
    # get the field and galaxy of this cluster
    field = row["field"]
    galaxy = row["galaxy"]

    # then get the needed quantities and store them
    dist = utils.distance(home_dir / "data" / field)
    dist_err = utils.distance_error(home_dir / "data" / field)
    row["galaxy_distance_mpc"] = dist.to("Mpc").value
    row["galaxy_distance_mpc_err"] = dist_err.to("Mpc").value
    row["galaxy_stellar_mass"] = gal_mass[galaxy]
    row["galaxy_sfr"] = gal_sfr[galaxy]

# then calculate specific star formation rate
catalog["galaxy_ssfr"] = catalog["galaxy_sfr"] / catalog["galaxy_stellar_mass"]

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
        "x_pix_snapshot_oversampled",
        "y_pix_snapshot_oversampled",
        "x_pix_snapshot_oversampled_e-",
        "x_pix_snapshot_oversampled_e+",
        "y_pix_snapshot_oversampled_e-",
        "y_pix_snapshot_oversampled_e+",
        "mag_F275W",
        "photoerr_F275W",
        "mag_F336W",
        "photoerr_F336W",
        "mag_F814W",
        "photoerr_F814W",
        "CI",
        "E(B-V)",
        "E(B-V)_max",
        "E(B-V)_min",
        "chi_2_F265W",
        "chi_2_F336W",
        "chi_2_F814W",
        "chi_2_reduced",
        "N_filters",
        "Q_probability",
        "dx_from_snap_center",
        "dy_from_snap_center",
    ]
)

# set the order for the leftover columns
new_col_order = [
    "field",
    "ID",
    "galaxy",
    "galaxy_distance_mpc",
    "galaxy_distance_mpc_err",
    "galaxy_stellar_mass",
    "galaxy_sfr",
    "galaxy_ssfr",
    "pixel_scale",
    "RA",
    "Dec",
    "x_pix_single",
    "y_pix_single",
    "morphology_class",
    "morphology_class_source",
    "age_yr",
    "age_yr_min",
    "age_yr_max",
    "mass_msun",
    "mass_msun_min",
    "mass_msun_max",
    "x_fitted",
    "x_fitted_e-",
    "x_fitted_e+",
    "y_fitted",
    "y_fitted_e-",
    "y_fitted_e+",
    "log_luminosity",
    "log_luminosity_e-",
    "log_luminosity_e+",
    "scale_radius_pixels",
    "scale_radius_pixels_e-",
    "scale_radius_pixels_e+",
    "axis_ratio",
    "axis_ratio_e-",
    "axis_ratio_e+",
    "position_angle",
    "position_angle_e-",
    "position_angle_e+",
    "power_law_slope",
    "power_law_slope_e-",
    "power_law_slope_e+",
    "local_background",
    "local_background_e-",
    "local_background_e+",
    "num_boostrapping_iterations",
    "radius_fit_failure",
    "profile_diff_reff",
    "reliable_radius",
    "reliable_mass",
    "r_eff_pixels",
    "r_eff_pixels_e-",
    "r_eff_pixels_e+",
    "r_eff_arcsec",
    "r_eff_arcsec_e-",
    "r_eff_arcsec_e+",
    "r_eff_pc",
    "r_eff_pc_e-",
    "r_eff_pc_e+",
    "crossing_time_yr",
    "crossing_time_log_err",
    "density",
    "density_log_err",
    "surface_density",
    "surface_density_log_err",
]

# check that I got all columns listed
assert len(new_col_order) == len(catalog.colnames)
assert sorted(new_col_order) == sorted(catalog.colnames)

# then apply this order
catalog = catalog[new_col_order]

# ======================================================================================
#
# Catalog validation
#
# ======================================================================================
# validate that there aren't nans or infinities where they don't belong
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

# validate that all clusters from the same galaxy have the same galaxy properties
for gal in np.unique(catalog["galaxy"]):
    mask = catalog["galaxy"] == gal
    assert len(np.unique(catalog["galaxy_distance_mpc"][mask])) == 1
    assert len(np.unique(catalog["galaxy_distance_mpc_err"][mask])) == 1
    assert len(np.unique(catalog["galaxy_stellar_mass"][mask])) == 1
    assert len(np.unique(catalog["galaxy_sfr"][mask])) == 1
    assert len(np.unique(catalog["galaxy_ssfr"][mask])) == 1
# double check the field distances too, since those are the same per field
for field in np.unique(catalog["field"]):
    mask = catalog["field"] == field
    assert len(np.unique(catalog["galaxy_distance_mpc"][mask])) == 1
    assert len(np.unique(catalog["galaxy_distance_mpc_err"][mask])) == 1

# ======================================================================================
#
# write the catalog!
#
# ======================================================================================
catalog.write(output_table, format="ascii.ecsv")
