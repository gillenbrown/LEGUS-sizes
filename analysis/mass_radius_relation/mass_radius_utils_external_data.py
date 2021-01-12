"""
Get external data sets as used in Krumholz et al 15, ARAA, 57, 227

I essentially copy the code used by Krumholz, which is publicly available

Each of these functions get the data from one source, and returns the following
quantities:
- mass
- mass error low
- mass error high
- radius
- radius error low
- radius error high
"""
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
def get_m31_open_clusters(max_age=1e9):
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
    fouesneau_14_age = 10 ** fouesneau_14_table[1].data["logA-best"]

    # then restrict to ones that have ids that work and ages in the appropriate range
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

        # check that there are no nans, and that the clusters aren't too old
        if (
            np.isnan(fouesneau_14_mass[fouesneau_14_idx])
            or np.isnan(johnson_12_r_eff_arcsec[johnson_12_idx])
            or fouesneau_14_age[fouesneau_14_idx] >= max_age
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
# Then the Ryon et al 15 M83 clusters
# ======================================================================================
def get_m83_clusters(max_age=1e9):
    hdulist = fits.open(data_path / "ryon2015_m83.fits")
    data = hdulist[1].data
    # Restrict to the clusters for which r_h is reliable, and which aren't too old
    mask = np.logical_and(data["eta"] > 1.3, data["logAge"] < np.log10(max_age))
    r_eff_log = data["logReff"][mask]
    r_eff_logerr = data["e_logReff"][mask]

    mass = 10 ** data["logMass"][mask]
    mass_min = 10 ** data["b_logMass"][mask]
    mass_max = 10 ** data["b_logmass_lc"][mask]

    # then convert the errors the linear min and max
    r_eff = 10 ** r_eff_log
    r_eff_err_hi = 10 ** (r_eff_log + r_eff_logerr) - r_eff
    r_eff_err_lo = r_eff - 10 ** (r_eff_log - r_eff_logerr)

    mass_err_hi = mass_max - mass
    mass_err_lo = mass - mass_min

    return mass, mass_err_lo, mass_err_hi, r_eff, r_eff_err_lo, r_eff_err_hi


# ======================================================================================
# Then young massive clusters from a few sources
# ======================================================================================
def get_mw_ymc_krumholz_19_clusters():
    # Young massive clusters in the Milky Way from our compilation
    data = table.Table.read(
        data_path / "mw_ymc_compilation.txt", format="ascii.basic", delimiter="\s"
    )
    mass_log = data["log_M"]
    r_eff = data["rh"]

    # errors need to be parsed, as not all clusters have them and the values are
    # strings. I convert to lists so I can edit the data type as I go
    mass_log_err = list(data["log_Merr"])
    r_eff_err = list(data["rh_err"])

    for col in [mass_log_err, r_eff_err]:
        for idx in range(len(col)):
            raw_value = col[idx]
            if raw_value == "--":
                value = 0
            else:
                value = float(raw_value)
            col[idx] = value

    # convert the log mass errors into regular errors
    mass = 10 ** mass_log
    mass_err_hi = 10 ** (mass_log + mass_log_err) - mass
    mass_err_lo = mass - 10 ** (mass_log - mass_log_err)

    return mass, mass_err_lo, mass_err_hi, r_eff, r_eff_err, r_eff_err


def get_m82_sscs():
    # Note that these have high masses (all above 1e5), so none of these are included
    # in the fit
    mass = (
        np.array(
            [40, 5.5, 3.9, 23, 4.0, 22, 2.7, 5.7, 7.3, 2.8, 2.7, 8.6, 5.2, 3.0, 2.5]
        )
        * 1e5
    )
    r_eff = np.array(
        [1.4, 1.5, 1.1, 2.5, 1.6, 2.7, 1.4, 3.0, 1.4, 1.9, 1.5, 2.1, 1.5, 1.7, 1.7]
    )
    # Errors are not present.
    all_err = np.zeros(len(mass))
    return mass, all_err, all_err, r_eff, all_err, all_err


def get_ngc253_sscs():
    mass = 10 ** np.array(
        [4.3, 4.3, 4.1, 5.0, 5.4, 5.3, 4.5, 4.8, 5.5, 5.3, 5.6, 6.0, 4.8, 5.5]
    )
    r_eff = (
        np.array([2.7, 1.2, 2.6, 2.5, 2.1, 2.1, 2.9, 1.9, 2.6, 3.5, 2.9, 4.3, 1.6, 1.9])
        / 2
    )
    # Errors are not present.
    all_err = np.zeros(len(mass))
    return mass, all_err, all_err, r_eff, all_err, all_err
