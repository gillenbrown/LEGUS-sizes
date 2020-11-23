"""
mass_size.py - plot the mass-size relation for LEGUS clusters
"""
import sys
from pathlib import Path

import numpy as np
from astropy import table
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
from scipy import optimize
from matplotlib import colors
import betterplotlib as bpl

# need to add the correct path to import utils
code_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(code_home_dir / "pipeline"))
import utils
import mass_radius_utils as mru

bpl.set_style()

# ======================================================================================
#
# load the parameters the user passed in
#
# ======================================================================================
plot_name = Path(sys.argv[1])
oversampling_factor = int(sys.argv[2])
psf_width = int(sys.argv[3])
psf_source = sys.argv[4]
catalogs = []
for item in sys.argv[5:]:
    this_cat = table.Table.read(item, format="ascii.ecsv")
    gal_dir = Path(item).parent.parent
    this_cat["distance"] = utils.distance(gal_dir).to("Mpc").value
    catalogs.append(this_cat)

# ======================================================================================
#
# Load galaxy data
#
# ======================================================================================
calzetti_path = plot_name.parent.parent / "analysis" / "calzetti_etal_15_table_1.txt"
galaxy_table = table.Table.read(
    calzetti_path, format="ascii.commented_header", header_start=3
)
# Add the data to my cluster tables
for cat in catalogs:
    cat["sfr"] = np.nan
    cat["m_star"] = np.nan
    cat["t_type"] = np.nan
    cat["ssfr"] = np.nan
    for row in cat:  # need to account for NGC 5194/NGC5195 in same catalog
        # throw away east-west-north-south field splits
        this_gal = row["galaxy"].split("-")[0].upper()
        for row_g in galaxy_table:
            if row_g["name"] == this_gal:
                row["sfr"] = row_g["sfr_uv_msun_per_year"]
                row["m_star"] = row_g["m_star"]
                row["t_type"] = row_g["morphology_t_type"]
                row["ssfr"] = row_g["sfr_uv_msun_per_year"] / row_g["m_star"]
                break

# then stack them together in one master catalog
big_catalog = table.vstack(catalogs, join_type="inner")

# ======================================================================================
#
# open the file to write fitting results to
#
# ======================================================================================
fit_out_file = open(plot_name.parent / "fits_table.txt", "w")
fit_out_file.write("\t\\begin{tabular}{lllll}\n")
fit_out_file.write("\t\t\\toprule\n")
fit_out_file.write(
    "\t\tSelection & "
    "$N$ & "
    "Slope & "
    "$\\reff$(pc) at $10^4\Msun$ & "
    "Intrinsic Scatter \\\\ \n"
)
fit_out_file.write("\t\t\midrule\n")

# ======================================================================================
#
# start parsing my catalogs
#
# ======================================================================================
# then filter out some clusters
print(f"Total Clusters: {len(big_catalog)}")
mask = big_catalog["good"]

# mask = np.logical_and(mask, big_catalog["age_yr"] <= 200e6)
# print(f"Clusters with age < 200 Myr: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["good"])
print(f"Clusters with good fits: {np.sum(mask)}")

mask = np.logical_and(mask, big_catalog["mass_msun"] > 0)
mask = np.logical_and(mask, big_catalog["mass_msun_max"] > 0)
mask = np.logical_and(mask, big_catalog["mass_msun_min"] > 0)
print(f"Clusters with good masses: {np.sum(mask)}")

ssfr_legus = big_catalog["ssfr"][mask]
distance_legus = big_catalog["distance"][mask]

age_legus = big_catalog["age_yr"][mask]
mass_legus = big_catalog["mass_msun"][mask]
# mass errors are reported as min and max values
mass_err_hi_legus = big_catalog["mass_msun_max"][mask] - mass_legus
mass_err_lo_legus = mass_legus - big_catalog["mass_msun_min"][mask]

r_eff_legus = big_catalog["r_eff_pc_rmax_15pix_best"][mask]
r_eff_err_hi_legus = big_catalog["r_eff_pc_rmax_15pix_e+"][mask]
r_eff_err_lo_legus = big_catalog["r_eff_pc_rmax_15pix_e-"][mask]

# Then transform them into log
log_mass_legus = np.log10(mass_legus)
log_mass_err_hi_legus = np.log10(mass_legus + mass_err_hi_legus) - log_mass_legus
log_mass_err_lo_legus = log_mass_legus - np.log10(mass_legus - mass_err_lo_legus)

log_r_eff_legus = np.log10(r_eff_legus)
log_r_eff_err_hi_legus = np.log10(r_eff_legus + r_eff_err_hi_legus) - log_r_eff_legus
log_r_eff_err_lo_legus = log_r_eff_legus - np.log10(r_eff_legus - r_eff_err_lo_legus)


# ======================================================================================
#
# Load cluster data from external sources - first is M31
#
# ======================================================================================
# M31 data. Masses and radii are in separate files.
data_path = code_home_dir / "analysis" / "krumholz_review_data"
johnson_12_table = fits.open(data_path / "johnson2012_m31.fits")
fouesneau_14_table = fits.open(data_path / "fouesneau2014_m31.fits")
# get the ids in both catalogs
johnson_12_id = johnson_12_table[1].data["PCID"]
fouesneau_14_id = fouesneau_14_table[1].data["PCID"]
m31_ids = np.intersect1d(johnson_12_id, fouesneau_14_id)

# get the relevant data
johnson_12_r_eff_arcsec = johnson_12_table[1].data["Reff"]
fouesneau_14_log_mass = fouesneau_14_table[1].data["logM-bset"]
fouesneau_14_log_mass_min = fouesneau_14_table[1].data["logM-p16"]
fouesneau_14_log_mass_max = fouesneau_14_table[1].data["logM-p84"]

# then restrict to ones that have ids that work
log_mass_m31 = []
log_mass_min_m31 = []
log_mass_max_m31 = []
r_eff_arcsec_m31 = []
for this_id in m31_ids:
    johnson_12_idx = np.where(johnson_12_id == this_id)[0]
    fouesneau_14_idx = np.where(fouesneau_14_id == this_id)[0]
    # numpy where gives arrays, we should only have one value here, make sure
    assert johnson_12_idx.size == 1
    assert fouesneau_14_idx.size == 1
    johnson_12_idx = johnson_12_idx[0]
    fouesneau_14_idx = fouesneau_14_idx[0]

    # check that there are no nans
    if np.isnan(fouesneau_14_log_mass[fouesneau_14_idx]) or np.isnan(
        johnson_12_r_eff_arcsec[fouesneau_14_idx]
    ):
        continue

    log_mass_m31.append(fouesneau_14_log_mass[fouesneau_14_idx])
    log_mass_min_m31.append(fouesneau_14_log_mass_min[fouesneau_14_idx])
    log_mass_max_m31.append(fouesneau_14_log_mass_max[fouesneau_14_idx])
    r_eff_arcsec_m31.append(johnson_12_r_eff_arcsec[fouesneau_14_idx])

log_mass_m31 = np.array(log_mass_m31)
log_mass_min_m31 = np.array(log_mass_min_m31)
log_mass_max_m31 = np.array(log_mass_max_m31)
r_eff_arcsec_m31 = np.array(r_eff_arcsec_m31)

# Johnson does not report errors on R_eff, so our only errors will be the distance
# errors
r_eff_full_m31 = utils.arcsec_to_pc_with_errors(Path("m31"), r_eff_arcsec_m31, 0, 0)
r_eff_m31, r_eff_err_lo_m31, r_eff_err_hi_m31 = r_eff_full_m31

# Then make both values have log and linear values for use in fitting and plotting
mass_m31 = 10 ** log_mass_m31
mass_err_lo_m31 = mass_m31 - 10 ** log_mass_min_m31
mass_err_hi_m31 = 10 ** log_mass_max_m31 - mass_m31

log_mass_err_lo_m31 = log_mass_m31 - log_mass_min_m31
log_mass_err_hi_m31 = log_mass_max_m31 - log_mass_m31

log_r_eff_m31 = np.log10(r_eff_m31)
log_r_eff_err_hi_m31 = np.log10(r_eff_m31 + r_eff_err_hi_m31) - log_r_eff_m31
log_r_eff_err_lo_m31 = log_r_eff_m31 - np.log10(r_eff_m31 - r_eff_err_lo_m31)

print(f"{len(mass_m31)} Clusters in M31")

# ======================================================================================
# Then the MW Open Clusters
# ======================================================================================
kharchenko_13_table = fits.open(data_path / "kharchenko2013_mw.fits")
kharchenko_13_mw_obj_type = kharchenko_13_table[1].data["Type"]
kharchenko_mw_dist = kharchenko_13_table[1].data["d"]
# restrict to solar neighborhood, not sure what the type does, but Krumholz uses it
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
r_eff_mw_ocs = kharchenko_mw_rc * np.interp(
    kharchenko_logc, kingtab.columns["logc"], kingtab.columns["xh2d"]
)
oort_A0 = 15.3 * u.km / (u.s * u.kpc)
oort_B0 = -11.9 * u.km / (u.s * u.kpc)
R0 = 8.2 * u.kpc
mw_rgc = np.sqrt(
    R0 ** 2
    + (kharchenko_mw_dist * u.pc) ** 2
    - 2.0 * R0 * kharchenko_mw_dist * u.pc * np.cos(kharchenko_mw_glon * np.pi / 180)
)
drg = (mw_rgc - R0) / R0
oort_A = oort_A0 * (1.0 - drg)
oort_A_minus_B = oort_A0 - oort_B0 - 2.0 * oort_A0 * drg
mass_mw_ocs = (
    (4.0 * oort_A * oort_A_minus_B * (kharchenko_mw_rt * u.pc) ** 3 / c.G)
    .to("Msun")
    .value
)

# Errors are not present.
r_eff_err_mw_ocs = np.zeros(r_eff_mw_ocs.shape)
log_r_eff_err_mw_ocs = np.zeros(r_eff_mw_ocs.shape)
mass_err_mw_ocs = np.zeros(r_eff_mw_ocs.shape)
log_mass_err_mw_ocs = np.zeros(r_eff_mw_ocs.shape)

# then convert this to linear space for plotting
log_r_eff_mw_ocs = np.log10(r_eff_mw_ocs)
log_mass_mw_ocs = np.log10(mass_mw_ocs)

print(f"{len(mass_mw_ocs)} Open Clusters in MW")

# ======================================================================================
# Then the MW Globular Clusters
# ======================================================================================
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
mass_mw_gcs = mw_gc_table["Mass"]
mass_err_mw_gcs = mw_gc_table["DM"]
r_eff_mw_gcs = mw_gc_table["rh,l"]
# no r_eff errors
r_eff_err_mw_gcs = np.zeros(r_eff_mw_gcs.shape)
log_r_eff_err_mw_gcs = np.zeros(r_eff_mw_gcs.shape)

# convert to log
log_mass_mw_gcs = np.log10(mass_mw_gcs)
log_r_eff_mw_gcs = np.log10(r_eff_mw_gcs)
log_mass_err_lo_mw_gcs = log_mass_mw_gcs - np.log10(mass_mw_gcs - mass_err_mw_gcs)
log_mass_err_hi_mw_gcs = np.log10(mass_mw_gcs + mass_err_mw_gcs) - log_mass_mw_gcs

print(f"{len(mass_mw_gcs)} Globular Clusters in MW")

# ======================================================================================
# Then the M31 Globular Clusters
# ======================================================================================
# Read Barmby+ 2007 table for M31 globulars
# m31_gc_tab = table.Table.read(data_path / "barmby2007_m31.txt", format="ascii.fixed_width")
# m31_gc_m = 10.**m31_gc_tab.columns['logM']
# m31_gc_Sigmah = 10.**m31_gc_tab.columns['logSh']
# m31_gc_rh = np.sqrt(m31_gc_m/m31_gc_Sigmah)


# ======================================================================================
#
# Then actually make different versions of this plot
#
# ======================================================================================
# First make the plot with all of my clusters
fit_legus, fit_legus_history = mru.fit_mass_size_relation(
    log_mass_legus,
    log_mass_err_lo_legus,
    log_mass_err_hi_legus,
    log_r_eff_legus,
    log_r_eff_err_lo_legus,
    log_r_eff_err_hi_legus,
    fit_mass_upper_limit=1e5,
)

fig, ax = bpl.subplots()
mru.plot_mass_size_dataset_scatter(
    ax,
    mass_legus,
    mass_err_lo_legus,
    mass_err_hi_legus,
    r_eff_legus,
    r_eff_err_lo_legus,
    r_eff_err_hi_legus,
    bpl.color_cycle[0],
)
mru.add_percentile_lines(ax, mass_legus, r_eff_legus)
mru.plot_best_fit_line(ax, fit_legus, 1e2, 1e5)
mru.format_mass_size_plot(ax)
fig.savefig(plot_name)
mru.write_fit_results(
    fit_out_file, "Full LEGUS Sample", len(r_eff_legus), fit_legus, fit_legus_history
)
# do another fit without old clusters
mask_not_old = age_legus < 1e9
fit_legus_young, fit_legus_young_history = mru.fit_mass_size_relation(
    log_mass_legus[mask_not_old],
    log_mass_err_lo_legus[mask_not_old],
    log_mass_err_hi_legus[mask_not_old],
    log_r_eff_legus[mask_not_old],
    log_r_eff_err_lo_legus[mask_not_old],
    log_r_eff_err_hi_legus[mask_not_old],
    fit_mass_upper_limit=1e5,
)
mru.write_fit_results(
    fit_out_file,
    "Age: 1 Myr - 1 Gyr",
    np.sum(mask_not_old),
    fit_legus_young,
    fit_legus_young_history,
)
mru.out_file_spacer(fit_out_file)

# --------------------------------------------------------------------------------------
# Then age split
# --------------------------------------------------------------------------------------
mask_young = age_legus < 1e7
mask_med = np.logical_and(age_legus >= 1e7, age_legus < 1e8)
mask_old = np.logical_and(age_legus >= 1e8, age_legus < 1e9)

fig, ax = bpl.subplots()
for mask, name, color, zorder in zip(
    [mask_young, mask_med, mask_old],
    ["Age: 1-10 Myr", "Age: 10-100 Myr", "Age: 100 Myr - 1 Gyr"],
    [bpl.color_cycle[0], bpl.color_cycle[5], bpl.color_cycle[3]],
    [1, 3, 2],
):
    fit_this, fit_this_history = mru.fit_mass_size_relation(
        log_mass_legus[mask],
        log_mass_err_lo_legus[mask],
        log_mass_err_hi_legus[mask],
        log_r_eff_legus[mask],
        log_r_eff_err_lo_legus[mask],
        log_r_eff_err_hi_legus[mask],
        fit_mass_upper_limit=1e5,
    )

    mru.plot_mass_size_dataset_contour(
        ax, mass_legus[mask], r_eff_legus[mask], color, zorder=zorder
    )
    # add_percentile_lines(ax, mass_legus[mask], r_eff_legus[mask], color=color)
    mru.plot_best_fit_line(
        ax, fit_this, 1, 1e5, color, fill=False, label=f"{name}, N={np.sum(mask)}"
    )
    mru.write_fit_results(fit_out_file, name, np.sum(mask), fit_this, fit_this_history)
mru.format_mass_size_plot(ax)
fig.savefig(plot_name.parent / "mass_size_relation_agesplit.pdf")
mru.out_file_spacer(fit_out_file)

# --------------------------------------------------------------------------------------
# Then SFH split
# --------------------------------------------------------------------------------------
cut_ssfr = 3e-10
mask_hi_ssfr = np.logical_and(ssfr_legus >= cut_ssfr, age_legus < 1e9)
mask_lo_ssfr = np.logical_and(ssfr_legus < cut_ssfr, age_legus < 1e9)

fig, ax = bpl.subplots()
for mask, name, color in zip(
    [mask_lo_ssfr, mask_hi_ssfr],
    [
        "sSFR $< 3 \\times 10^{-10} {\\rm yr}^{-1}$",
        "sSFR $\geq 3 \\times 10^{-10} {\\rm yr}^{-1}$",
    ],
    [bpl.color_cycle[3], bpl.color_cycle[0]],
):
    fit_this, fit_this_history = mru.fit_mass_size_relation(
        log_mass_legus[mask],
        log_mass_err_lo_legus[mask],
        log_mass_err_hi_legus[mask],
        log_r_eff_legus[mask],
        log_r_eff_err_lo_legus[mask],
        log_r_eff_err_hi_legus[mask],
        fit_mass_upper_limit=1e5,
    )

    mru.plot_mass_size_dataset_contour(
        ax,
        mass_legus[mask],
        r_eff_legus[mask],
        color,
    )
    # add_percentile_lines(ax, mass_legus[mask], r_eff_legus[mask], color=color)
    mru.plot_best_fit_line(
        ax, fit_this, 1, 1e5, color, fill=False, label=f"{name}, N={np.sum(mask)}"
    )
    mru.write_fit_results(fit_out_file, name, np.sum(mask), fit_this, fit_this_history)
mru.format_mass_size_plot(ax)
fig.savefig(plot_name.parent / "mass_size_relation_ssfrsplit.pdf")
mru.out_file_spacer(fit_out_file)

# --------------------------------------------------------------------------------------
# Then distance split
# --------------------------------------------------------------------------------------
mask_dist_1 = np.logical_and(
    np.logical_and(distance_legus >= 3, distance_legus < 5), age_legus < 1e9
)
mask_dist_2 = np.logical_and(
    np.logical_and(distance_legus >= 7, distance_legus < 9), age_legus < 1e9
)


fig, ax = bpl.subplots()
for mask, name, color in zip(
    [mask_dist_1, mask_dist_2],
    ["Distance: 3-5 Mpc", "Distance: 7-9 Mpc"],
    [bpl.color_cycle[2], bpl.color_cycle[7]],
):
    fit_this, fit_this_history = mru.fit_mass_size_relation(
        log_mass_legus[mask],
        log_mass_err_lo_legus[mask],
        log_mass_err_hi_legus[mask],
        log_r_eff_legus[mask],
        log_r_eff_err_lo_legus[mask],
        log_r_eff_err_hi_legus[mask],
        fit_mass_upper_limit=1e5,
    )

    mru.plot_mass_size_dataset_contour(
        ax,
        mass_legus[mask],
        r_eff_legus[mask],
        color,
    )
    # add_percentile_lines(ax, mass_legus[mask], r_eff_legus[mask], color=color)
    mru.plot_best_fit_line(
        ax, fit_this, 1, 1e5, color, fill=False, label=f"{name}, N={np.sum(mask)}"
    )
    mru.write_fit_results(fit_out_file, name, np.sum(mask), fit_this, fit_this_history)
mru.format_mass_size_plot(ax, xmax=3e5)
fig.savefig(plot_name.parent / "mass_size_relation_distancesplit.pdf")
mru.out_file_spacer(fit_out_file)

# --------------------------------------------------------------------------------------
# Then adding different datasets
# --------------------------------------------------------------------------------------
fit_legus_m31, fit_legus_m31_history = mru.fit_mass_size_relation(
    np.concatenate([log_mass_legus, log_mass_m31]),
    np.concatenate([log_mass_err_lo_legus, log_mass_err_lo_m31]),
    np.concatenate([log_mass_err_hi_legus, log_mass_err_hi_m31]),
    np.concatenate([log_r_eff_legus, log_r_eff_m31]),
    np.concatenate([log_r_eff_err_lo_legus, log_r_eff_err_lo_m31]),
    np.concatenate([log_r_eff_err_hi_legus, log_r_eff_err_hi_m31]),
    fit_mass_upper_limit=1e5,
)
mru.write_fit_results(
    fit_out_file,
    "LEGUS + M31",
    len(log_mass_legus) + len(log_mass_m31),
    fit_legus_m31,
    fit_legus_m31_history,
)

fit_legus_mw, fit_legus_mw_history = mru.fit_mass_size_relation(
    np.concatenate([log_mass_legus, log_mass_mw_ocs]),
    np.concatenate([log_mass_err_lo_legus, log_mass_err_mw_ocs]),
    np.concatenate([log_mass_err_hi_legus, log_mass_err_mw_ocs]),
    np.concatenate([log_r_eff_legus, log_r_eff_mw_ocs]),
    np.concatenate([log_r_eff_err_lo_legus, log_r_eff_err_mw_ocs]),
    np.concatenate([log_r_eff_err_hi_legus, log_r_eff_err_mw_ocs]),
    fit_mass_upper_limit=1e5,
)
mru.write_fit_results(
    fit_out_file,
    "LEGUS + MW",
    len(log_mass_legus) + len(log_mass_mw_ocs),
    fit_legus_mw,
    fit_legus_mw_history,
)

# --------------------------------------------------------------------------------------
# Then all datasets, with a plot
# --------------------------------------------------------------------------------------
fit_legus_mw_m31, fit_legus_mw_m31_history = mru.fit_mass_size_relation(
    np.concatenate([log_mass_legus, log_mass_m31, log_mass_mw_ocs]),
    np.concatenate(
        [
            log_mass_err_lo_legus,
            log_mass_err_lo_m31,
            log_mass_err_mw_ocs,
        ]
    ),
    np.concatenate(
        [
            log_mass_err_hi_legus,
            log_mass_err_hi_m31,
            log_mass_err_mw_ocs,
        ]
    ),
    np.concatenate([log_r_eff_legus, log_r_eff_m31, log_r_eff_mw_ocs]),
    np.concatenate(
        [
            log_r_eff_err_lo_legus,
            log_r_eff_err_lo_m31,
            log_r_eff_err_mw_ocs,
        ]
    ),
    np.concatenate(
        [
            log_r_eff_err_hi_legus,
            log_r_eff_err_hi_m31,
            log_r_eff_err_mw_ocs,
        ]
    ),
    fit_mass_upper_limit=1e5,
)
fig, ax = bpl.subplots()
mru.plot_mass_size_dataset_scatter(
    ax,
    mass_legus,
    mass_err_lo_legus,
    mass_err_hi_legus,
    r_eff_legus,
    r_eff_err_lo_legus,
    r_eff_err_hi_legus,
    bpl.color_cycle[0],
    "LEGUS",
)
mru.plot_mass_size_dataset_scatter(
    ax,
    mass_m31,
    mass_err_lo_m31,
    mass_err_hi_m31,
    r_eff_m31,
    r_eff_err_lo_m31,
    r_eff_err_hi_m31,
    bpl.color_cycle[3],
    "M31",
)
mru.plot_mass_size_dataset_scatter(
    ax,
    mass_mw_ocs,
    mass_err_mw_ocs,
    mass_err_mw_ocs,
    r_eff_mw_ocs,
    r_eff_err_mw_ocs,
    r_eff_err_mw_ocs,
    bpl.color_cycle[4],
    "MW Open Clusters",
)

mru.add_percentile_lines(
    ax,
    np.concatenate([mass_legus, mass_m31, mass_mw_ocs]),
    np.concatenate([r_eff_legus, r_eff_m31, r_eff_mw_ocs]),
)
# add_percentile_lines(ax, mass_m31, r_eff_m31, style="unique")
mru.plot_best_fit_line(ax, fit_legus_mw_m31, 1, 1e5)
# add_psfs_to_plot(ax, x_max=1e7)
mru.format_mass_size_plot(ax, xmin=1, xmax=1e7)
mru.write_fit_results(
    fit_out_file,
    "LEGUS + M31 + MW",
    len(mass_m31) + len(mass_legus) + len(mass_mw_ocs),
    fit_legus_mw_m31,
    fit_legus_mw_m31_history,
)
fig.savefig(plot_name.parent / "mass_size_legus_m31_mw.pdf")

# ======================================================================================
#
# finalize output file
#
# ======================================================================================
fit_out_file.write("\t\t\\bottomrule\n")
fit_out_file.write("\t\end{tabular}\n")
fit_out_file.close()
