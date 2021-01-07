from pathlib import Path
import sys

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from astropy import table

# need to add the correct path to import utils
code_home_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(code_home_dir / "pipeline"))
import utils

# define the directory where everything is stored
data_path = code_home_dir / "analysis" / "krumholz_review_data"

# ======================================================================================
# M31 open clusters
# ======================================================================================
def get_m31_open_clusters():
    # M31 data. Masses and radii are in separate files.
    johnson_12_table = fits.open(data_path / "johnson2012_m31.fits")
    fouesneau_14_table = fits.open(data_path / "fouesneau2014_m31.fits")
    # get the ids in both catalogs
    johnson_12_id = johnson_12_table[1].data["PCID"]
    fouesneau_14_id = fouesneau_14_table[1].data["PCID"]
    m31_ids = np.intersect1d(johnson_12_id, fouesneau_14_id)

    # get the relevant data
    johnson_12_r_eff_arcsec = johnson_12_table[1].data["Reff"]
    fouesneau_14_mass = 10 ** fouesneau_14_table[1].data["logM-bset"]
    fouesneau_14_mass_min = 10 ** fouesneau_14_table[1].data["logM-p16"]
    fouesneau_14_mass_max = 10 ** fouesneau_14_table[1].data["logM-p84"]

    # then restrict to ones that have ids that work
    mass = []
    mass_min = []
    mass_max = []
    r_eff_arcsec = []
    for this_id in m31_ids:
        johnson_12_idx = np.where(johnson_12_id == this_id)[0]
        fouesneau_14_idx = np.where(fouesneau_14_id == this_id)[0]
        # numpy where gives arrays, we should only have one value here, make sure
        assert johnson_12_idx.size == 1
        assert fouesneau_14_idx.size == 1
        johnson_12_idx = johnson_12_idx[0]
        fouesneau_14_idx = fouesneau_14_idx[0]

        # check that there are no nans
        if np.isnan(fouesneau_14_mass[fouesneau_14_idx]) or np.isnan(
            johnson_12_r_eff_arcsec[johnson_12_idx]
        ):
            continue

        mass.append(fouesneau_14_mass[fouesneau_14_idx])
        mass_min.append(fouesneau_14_mass_min[fouesneau_14_idx])
        mass_max.append(fouesneau_14_mass_max[fouesneau_14_idx])
        r_eff_arcsec.append(johnson_12_r_eff_arcsec[johnson_12_idx])

    mass = np.array(mass)
    mass_min = np.array(mass_min)
    mass_max = np.array(mass_max)
    r_eff_arcsec = np.array(r_eff_arcsec)

    # Johnson does not report errors on R_eff, so our only errors will be the distance
    # errors
    r_eff_full = utils.arcsec_to_pc_with_errors(Path("m31"), r_eff_arcsec, 0, 0)
    r_eff, r_eff_err_lo, r_eff_err_hi = r_eff_full

    # then turn limits into errors
    mass_err_lo = mass - mass_min
    mass_err_hi = mass_max - mass

    return mass, mass_err_lo, mass_err_hi, r_eff, r_eff_err_lo, r_eff_err_hi


# ======================================================================================
# Then the MW Open Clusters
# ======================================================================================
def get_mw_open_clusters():
    kharchenko_13_table = fits.open(data_path / "kharchenko2013_mw.fits")
    kharchenko_13_mw_obj_type = kharchenko_13_table[1].data["Type"]
    kharchenko_mw_dist = kharchenko_13_table[1].data["d"]
    # restrict to solar neighborhood, not sure what the type does, but
    # Krumholz uses it
    kharchenko_good_idx = np.logical_and(
        [str(o_type) != "g" for o_type in kharchenko_13_mw_obj_type],
        kharchenko_mw_dist <= 2e3,
    )
    kharchenko_mw_Sigma = kharchenko_13_table[1].data["k"][kharchenko_good_idx]
    kharchenko_mw_rt = kharchenko_13_table[1].data["rt"][kharchenko_good_idx]
    kharchenko_mw_rc = kharchenko_13_table[1].data["rc"][kharchenko_good_idx]
    kharchenko_mw_k = kharchenko_13_table[1].data["k"][kharchenko_good_idx]
    kharchenko_mw_glat = kharchenko_13_table[1].data["GLAT"][kharchenko_good_idx]
    kharchenko_mw_glon = kharchenko_13_table[1].data["GLON"][kharchenko_good_idx]
    kharchenko_mw_dist = kharchenko_mw_dist[kharchenko_good_idx]

    # Following code copied from Krumholz:
    # Convert Kharchenko's King profile r_t and r_c measurements into
    # projected half-mass / half-number radii and mass; mass is
    # derived following equation (3) of Piskunov+ 2007, A&A, 468, 151,
    # using updated values of the Oort constants from Bovy+ 2017, MNRAS,
    # 468, L63, and the Sun's distance from the Galactic Center from
    # Bland-Hawthorn & Gerhard, 2016, ARA&A, 54, 529; note that this
    # calculation implicitly assumes that the Sun lies in the galactic
    # plane, which is not exactly true (z0 ~= 25 pc), but the error
    # associated with this approximation is small compared to the
    # uncertainty in the distance to the Galctic Centre
    kingtab = table.Table.read(data_path / "kingtab.txt", format="ascii")
    kharchenko_logc = np.log10(kharchenko_mw_rt / kharchenko_mw_rc)
    r_eff = kharchenko_mw_rc * np.interp(
        kharchenko_logc, kingtab.columns["logc"], kingtab.columns["xh2d"]
    )
    oort_A0 = 15.3 * u.km / (u.s * u.kpc)
    oort_B0 = -11.9 * u.km / (u.s * u.kpc)
    R0 = 8.2 * u.kpc
    mw_rgc = np.sqrt(
        R0 ** 2
        + (kharchenko_mw_dist * u.pc) ** 2
        - 2.0
        * R0
        * kharchenko_mw_dist
        * u.pc
        * np.cos(kharchenko_mw_glon * np.pi / 180)
    )
    drg = (mw_rgc - R0) / R0
    oort_A = oort_A0 * (1.0 - drg)
    oort_A_minus_B = oort_A0 - oort_B0 - 2.0 * oort_A0 * drg
    mass = (
        (4.0 * oort_A * oort_A_minus_B * (kharchenko_mw_rt * u.pc) ** 3 / c.G)
        .to("Msun")
        .value
    )

    # some of these have nans, so throw them out
    mask = ~np.isnan(mass)
    mass = mass[mask]
    r_eff = r_eff[mask]

    # Errors are not present.
    all_err = np.zeros(len(mass))

    return mass, all_err, all_err, r_eff, all_err, all_err


# ======================================================================================
# Then the MW Globular Clusters
# ======================================================================================
def get_mw_globular_clusters():
    # Read Baumgardt & Hilker 2018 table for MW globulars
    mw_gc_table = table.Table.read(
        data_path / "baumgardt2018.txt",
        format="ascii.fixed_width",
        # fmt: off
        col_starts=[
            0, 9, 20, 31, 38, 46, 52, 61, 70, 81, 92, 100, 107, 114,
            121, 128, 136, 145, 153, 161, 169, 176, 183, 190, 198
        ],
        # fmt: on
        header_start=0,
        data_start=2,
    )
    mass = mw_gc_table["Mass"]
    mass_errs = mw_gc_table["DM"]
    r_eff = mw_gc_table["rh,l"]
    # no r_eff errors
    r_eff_err = np.zeros(r_eff.shape)

    return mass, mass_errs, mass_errs, r_eff, r_eff_err, r_eff_err
