import sys
from pathlib import Path

from astropy import table
import numpy as np
import betterplotlib as bpl
from sinistra.astropy_helpers import symmetric_match

bpl.set_style()

# ======================================================================================
#
# Load the various catalogs
#
# ======================================================================================
ryon_628 = table.Table.read("ryon_results_ngc628.txt", format="ascii.cds")
ryon_1313 = table.Table.read("ryon_results_ngc1313.txt", format="ascii.cds")
# then make new columns, since Ryon's tables are in log radius, and I just want radius
for cat in [ryon_628, ryon_1313]:
    cat["r_eff_Galfit"] = 10 ** cat["logReff-gal"]
    cat["e_r_eff+_Galfit"] = np.minimum(
        10 ** (cat["logReff-gal"] + cat["E_logReff-gal"]) - cat["r_eff_Galfit"], 1000
    )
    cat["e_r_eff-_Galfit"] = np.minimum(
        cat["r_eff_Galfit"] - 10 ** (cat["logReff-gal"] - cat["e_logReff-gal"]), 100
    )

    cat["r_eff_CI"] = 10 ** cat["logReff-ci"]
    cat["e_r_eff+_CI"] = (
        10 ** (cat["logReff-ci"] + cat["E_logReff-ci"]) - cat["r_eff_CI"]
    )
    cat["e_r_eff-_CI"] = cat["r_eff_CI"] - 10 ** (
        cat["logReff-ci"] - cat["e_logReff-ci"]
    )

# Go through all the cluster catalogs that are passed in, and get the ones we need
catalogs = {"ngc1313-e": None, "ngc1313-w": None, "ngc628-c": None, "ngc628-e": None}
for item in sys.argv[1:]:
    path = Path(item)
    galaxy_name = path.parent.parent.name
    if galaxy_name in catalogs:
        catalogs[galaxy_name] = table.Table.read(item, format="hdf5")

# check that none of these are empty
for key in catalogs:
    if catalogs[key] is None:
        raise RuntimeError(f"The {key} catalog has not been created!")

# ======================================================================================
#
# Matching catalogs
#
# ======================================================================================
# Here Ryon's catalogs combine the two fields for these two galaxies. I won't do that,
# as I want to compare the different fields separately, but I do need to match to the
# IDs used in the Ryon catalogs
for name, cat in catalogs.items():
    cat["ID"] = [f"{i}{name[-2:]}" for i in cat["ID"]]

# Then match the tables together. Using the inner join type is the strict intersection
# where the matched keys must match exactly
matches = dict()
matches["ngc1313-e"] = table.join(
    catalogs["ngc1313-e"], ryon_1313, join_type="inner", keys="ID"
)
matches["ngc1313-w"] = table.join(
    catalogs["ngc1313-w"], ryon_1313, join_type="inner", keys="ID"
)
matches["ngc628-e"] = table.join(
    catalogs["ngc628-e"], ryon_628, join_type="inner", keys="ID"
)
# The NGC 628 Center field has different IDs than the published tables! I need to match
# based on RA/Dec
matches["ngc628-c"] = symmetric_match(
    catalogs["ngc628-c"],
    ryon_628,
    ra_col_1="RA",
    ra_col_2="RAdeg",
    dec_col_1="Dec",
    dec_col_2="DEdeg",
    max_sep=0.03,
)

# ======================================================================================
#
# Calculate RMS
#
# ======================================================================================
total_rms = 0
num_clusters = 0

for field, cat in matches.items():
    for row in cat:
        if row["Eta"] > 1.3 and row["power_law_slope_best"] > 1.3:
            my_r_eff = row["r_eff_pc_no_rmax_best"]
            ryon_r_eff = row["r_eff_Galfit"]
            if my_r_eff > ryon_r_eff:
                ryon_err = row["e_r_eff+_Galfit"]
                my_err = row["r_eff_pc_no_rmax_e-"]
            else:
                ryon_err = row["e_r_eff-_Galfit"]
                my_err = row["r_eff_pc_no_rmax_e+"]

            used_err = max(ryon_err, my_err)
            diff = (my_r_eff - ryon_r_eff) / used_err
            if diff < 30:
                total_rms += ((my_r_eff - ryon_r_eff) / used_err) ** 2
                num_clusters += 1

normalized_rms = np.sqrt(total_rms / num_clusters)

# ======================================================================================
#
# Making plots
#
# ======================================================================================
limits = 0.3, 20
# First we'll make a straight comparison
fig, ax = bpl.subplots(figsize=[7, 7])
for idx, (field, cat) in enumerate(matches.items()):
    ryon_eta = cat["Eta"]
    # my_eta = cat["power_law_slope_best"]
    ryon_mask = ryon_eta > 1.3
    # my_mask = my_eta > 1.3
    mask = ryon_mask  # np.logical_and(ryon_mask, my_mask)

    c = bpl.color_cycle[idx]

    ax.errorbar(
        x=cat["r_eff_Galfit"][mask],
        y=cat["r_eff_pc_no_rmax_median"][mask],
        xerr=[cat["e_r_eff-_Galfit"][mask], cat["e_r_eff+_Galfit"][mask]],
        yerr=[cat["r_eff_pc_no_rmax_e-"][mask], cat["r_eff_pc_no_rmax_e+"][mask]],
        markerfacecolor=c,
        markeredgecolor=c,
        markersize=5,
        ecolor=c,
        label=field.upper(),
        elinewidth=0.5,
        zorder=2,
    )

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_limits(*limits, *limits)
ax.plot(limits, limits, c=bpl.almost_black, lw=1, zorder=0)
ax.equal_scale()
ax.legend(loc=4)
ax.easy_add_text(f"RMS = {normalized_rms:.3f}", "upper left")
ax.add_labels("Cluster $R_{eff}$ [pc] - Ryon+ 2017", "Cluster $R_{eff}$ [pc] - Me")
fig.savefig("comparison_plot.png")
