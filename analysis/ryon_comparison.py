import sys
from pathlib import Path

from astropy import table
import numpy as np
import betterplotlib as bpl
from sinistra.astropy_helpers import symmetric_match

bpl.set_style()

# need to add the correct path to import utils
code_home_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(code_home_dir / "pipeline"))
import utils

plot_name = Path(sys.argv[1])
full_ryon = sys.argv[2].lower() == "ryon"

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
for item in sys.argv[3:]:
    path = Path(item)
    data_dir = path.parent.parent
    galaxy_name = data_dir.name
    if galaxy_name in catalogs:
        this_cat = table.Table.read(item, format="ascii.ecsv")

        # If we need to, modify the radii so they use the same distances as ryon
        if not full_ryon:
            # scale the radii by the ratio of the distances as used in Ryon and as
            # used in my work
            distance_ryon = utils.distance(data_dir, True).to("Mpc").value
            distance_me = utils.distance(data_dir, False).to("Mpc").value
            dist_factor = distance_ryon / distance_me
            this_cat["r_eff_pc_rmax_15pix_best"] *= dist_factor
            this_cat["r_eff_pc_rmax_15pix_e-"] *= dist_factor
            this_cat["r_eff_pc_rmax_15pix_e+"] *= dist_factor

        catalogs[galaxy_name] = this_cat

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
    ryon_mask_good = cat["Eta"] >= 1.3
    # section 4.1 of Ryon+17 lists the number of clusters that pass the eta cut:
    # NGC1313-e: 14
    # NGC1313-w: 45
    # NGC628-c: 107
    # NGC628-e: 27
    # using print statements here I get the same thing
    # if we're using my full method, I don't need to do my restriction on slope
    my_mask_good = cat["good_radius"]
    if full_ryon:
        my_mask_good = np.logical_and(cat["power_law_slope_best"] >= 1.3, my_mask_good)

    cat["good_for_plot"] = np.logical_and(ryon_mask_good, my_mask_good)

for field, cat in matches.items():
    for row in cat:
        if row["good_for_plot"]:
            my_r_eff = row["r_eff_pc_rmax_15pix_best"]
            ryon_r_eff = row["r_eff_Galfit"]
            if my_r_eff > ryon_r_eff:
                ryon_err = row["e_r_eff+_Galfit"]
                my_err = row["r_eff_pc_rmax_15pix_e-"]
            else:
                ryon_err = row["e_r_eff-_Galfit"]
                my_err = row["r_eff_pc_rmax_15pix_e+"]
            used_err = np.sqrt(ryon_err ** 2 + my_err ** 2)
            diff = (my_r_eff - ryon_r_eff) / used_err
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
    mask = cat["good_for_plot"]

    c = bpl.color_cycle[idx]

    ax.errorbar(
        x=cat["r_eff_Galfit"][mask],
        y=cat["r_eff_pc_rmax_15pix_best"][mask],
        xerr=[cat["e_r_eff-_Galfit"][mask], cat["e_r_eff+_Galfit"][mask]],
        yerr=[
            cat["r_eff_pc_rmax_15pix_e-"][mask],
            cat["r_eff_pc_rmax_15pix_e+"][mask],
        ],
        markerfacecolor=c,
        markeredgecolor=c,
        markersize=5,
        ecolor=c,
        label=field.replace("ngc", "NGC "),
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
ax.add_labels(
    "Cluster $R_{eff}$ [pc] - Ryon+ 2017", "Cluster $R_{eff}$ [pc] - This Work"
)
fig.savefig(plot_name)
