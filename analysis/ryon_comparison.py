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

# ======================================================================================
#
# Load the various catalogs
#
# ======================================================================================
ryon_628 = table.Table.read("ryon_results_ngc628.txt", format="ascii.cds")
ryon_1313 = table.Table.read("ryon_results_ngc1313.txt", format="ascii.cds")
# then make new columns, since Ryon's tables are in log radius, and I just want radius
for cat in [ryon_628, ryon_1313]:
    cat["r_eff_ryon"] = 10 ** cat["logReff-gal"]
    cat["r_eff_e+_ryon"] = (
        10 ** (cat["logReff-gal"] + cat["E_logReff-gal"]) - cat["r_eff_ryon"]
    )
    cat["r_eff_e-_ryon"] = cat["r_eff_ryon"] - 10 ** (
        cat["logReff-gal"] - cat["e_logReff-gal"]
    )

# Go through all the cluster catalogs that are passed in, and get the ones we need
# f stands for my full method, r is the ryon-like
catalogs_f = {"ngc1313-e": None, "ngc1313-w": None, "ngc628-c": None, "ngc628-e": None}
catalogs_r = catalogs_f.copy()
# I have both ryon-like and non-ryon-like catalogs. I pass both in using a very long
# list of catalogs.
for item in sys.argv[2:]:
    path = Path(item)
    data_dir = path.parent.parent
    galaxy_name = data_dir.name
    if galaxy_name in catalogs_f:
        # See if this will be a ryon-like catalog or not
        if "ryonlike" in item:
            suffix = "ryonlike"
        else:
            suffix = "full"

        this_cat = table.Table.read(item, format="ascii.ecsv")

        # rename the r_eff columns to be easier to use. Also rename the eta column,
        # since I'll use that later too
        this_cat.rename_column("r_eff_pc_rmax_15pix_best", "r_eff")
        this_cat.rename_column("r_eff_pc_rmax_15pix_e-", "r_eff_e-")
        this_cat.rename_column("r_eff_pc_rmax_15pix_e+", "r_eff_e+")
        this_cat.rename_column("power_law_slope_best", "eta")

        # modify the radii so they use the same distances as ryon
        # scale the radii by the ratio of the distances as used in Ryon and as
        # used in my work. The ryon-like catalogs have already done this, so there
        # is nothing needed there.
        if suffix == "full":
            distance_ryon = utils.distance(data_dir, True).to("Mpc").value
            distance_me = utils.distance(data_dir, False).to("Mpc").value
            dist_factor = distance_ryon / distance_me
            this_cat["r_eff"] *= dist_factor
            this_cat["r_eff_e-"] *= dist_factor
            this_cat["r_eff_e+"] *= dist_factor

        # rename all the columns to either be "ryonlike" or "full", depending on
        # which this catalog is. This will avoid overlap when matching
        for col in this_cat.colnames:
            if col == "ID":  # don't rename ID
                continue
            this_cat.rename_column(col, col + "_" + suffix)

        # Then store these catalogs
        if suffix == "ryonlike":
            catalogs_r[galaxy_name] = this_cat
        else:
            catalogs_f[galaxy_name] = this_cat

# check that none of these are empty
for key in catalogs_f:
    if catalogs_r[key] is None:
        raise RuntimeError(f"The {key} Ryon-like catalog has not been created!")
    if catalogs_f[key] is None:
        raise RuntimeError(f"The {key} catalog has not been created!")

# ======================================================================================
#
# Matching catalogs
#
# ======================================================================================
# Here Ryon's catalogs combine the two fields for these two galaxies. I won't do that,
# as I want to compare the different fields separately, but I do need to match to the
# format of IDs used in the Ryon catalogs
for name, cat in catalogs_f.items():
    cat["ID"] = [f"{i}{name[-2:]}" for i in cat["ID"]]
for name, cat in catalogs_r.items():
    cat["ID"] = [f"{i}{name[-2:]}" for i in cat["ID"]]

# First I'll match my catalogs together, then afterwards I'll match that to Ryon's
# Using the inner join type is the strict intersection where the matched keys must
# match exactly
common = {"join_type": "inner", "keys": "ID"}
matches = dict()
for field in catalogs_f:
    matches[field] = table.join(catalogs_f[field], catalogs_r[field], **common)

# Then match to Ryon.
matches["ngc1313-e"] = table.join(matches["ngc1313-e"], ryon_1313, **common)
matches["ngc1313-w"] = table.join(matches["ngc1313-w"], ryon_1313, **common)
matches["ngc628-e"] = table.join(matches["ngc628-e"], ryon_628, **common)
# The NGC 628 Center field has different IDs than the published tables! I need to match
# based on RA/Dec
matches["ngc628-c"] = symmetric_match(
    matches["ngc628-c"],
    ryon_628,
    ra_col_1="RA_full",
    ra_col_2="RAdeg",
    dec_col_1="Dec_full",
    dec_col_2="DEdeg",
    max_sep=0.03,
)

# ======================================================================================
#
# masking
#
# ======================================================================================
for field, cat in matches.items():
    ryon_mask = cat["Eta"] >= 1.3
    # section 4.1 of Ryon+17 lists the number of clusters that pass the eta cut:
    # NGC1313-e: 14
    # NGC1313-w: 45
    # NGC628-c: 107
    # NGC628-e: 27
    # using print statements here I get the same thing
    # print(field, np.sum(ryon_mask))

    # in this comparison with my ryon-like method I need the fit to be successfull and
    # for eta>1.3 here aws well.
    cat["mask_ryon_ryonlike"] = np.logical_and.reduce(
        [
            ryon_mask,
            cat["good_radius_ryonlike"],
            cat["eta_ryonlike"] >= 1.3,
        ]
    )

    # for the comparison of my two methods, I still need the restriction on eta for the
    # ryon-like fit, and both of my fits need to be successfull.
    cat["mask_ryonlike_full"] = np.logical_and.reduce(
        [
            cat["good_radius_full"],
            cat["good_radius_ryonlike"],
            cat["eta_ryonlike"] > 1.3,
        ]
    )

    # then one to compare my full method to ryon's. I don't need my method to have
    # eta > 1.3, but Ryon still needs that.
    cat["mask_ryon_full"] = np.logical_and.reduce(
        [
            ryon_mask,
            cat["good_radius_full"],
        ]
    )

# ======================================================================================
#
# Calculate RMS
#
# ======================================================================================
def rms(suffix_1, suffix_2, mask_col_name):
    sum_squares = 0
    num_clusters = 0

    for field, cat in matches.items():
        for row in cat:
            if row[mask_col_name]:
                r_eff_1 = row[f"r_eff_{suffix_1}"]
                r_eff_2 = row[f"r_eff_{suffix_2}"]

                # use the appropriate asymmetric error
                if r_eff_1 > r_eff_2:
                    err_1 = row[f"r_eff_e-_{suffix_1}"]
                    err_2 = row[f"r_eff_e+_{suffix_2}"]
                else:
                    err_1 = row[f"r_eff_e+_{suffix_1}"]
                    err_2 = row[f"r_eff_e-_{suffix_2}"]
                # add the errors in quadrature
                err = np.sqrt(err_1 ** 2 + err_2 ** 2)

                # then compile the sum of squares
                sum_squares += ((r_eff_1 - r_eff_2) / err) ** 2
                num_clusters += 1

    rms = np.sqrt(sum_squares / num_clusters)
    return rms


# ======================================================================================
#
# Making plots
#
# ======================================================================================
limits = 0.3, 20
# First we'll make a straight comparison
fig, axs = bpl.subplots(ncols=2, figsize=[14, 7])

colors = {
    "ngc1313-e": bpl.color_cycle[0],
    "ngc1313-w": bpl.color_cycle[1],
    "ngc628-e": bpl.color_cycle[2],
    "ngc628-c": bpl.color_cycle[3],
}
labels = {
    "ryon": "$R_{eff}$ [pc] - Ryon+ 2017",
    "ryonlike": "$R_{eff}$ [pc] - This Work Ryon-like",
    "full": "$R_{eff}$ [pc] - This Work Full Method",
}
# In the left panel, we compare my Ryon-like (x) to Ryon's results (y). Then in the
# right panel I compare my two methods. I wrote this to be flexible though.
for ax, x_suffix, y_suffix, mask_col_name in zip(
    axs,
    ["ryonlike", "ryonlike"],
    ["ryon", "full"],
    ["mask_ryon_ryonlike", "mask_ryonlike_full"],
):
    for field, cat in matches.items():
        mask = cat[mask_col_name]

        c = colors[field]

        ax.errorbar(
            x=cat[f"r_eff_{x_suffix}"][mask],
            y=cat[f"r_eff_{y_suffix}"][mask],
            xerr=[cat[f"r_eff_e-_{x_suffix}"][mask], cat[f"r_eff_e+_{x_suffix}"][mask]],
            yerr=[cat[f"r_eff_e-_{y_suffix}"][mask], cat[f"r_eff_e+_{y_suffix}"][mask]],
            markerfacecolor=c,
            markeredgecolor=c,
            markersize=5,
            ecolor=c,
            label=field.replace("ngc", "NGC "),
            elinewidth=0.5,
            zorder=2,
        )

    print(y_suffix, rms(x_suffix, y_suffix, mask_col_name))

    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_limits(*limits, *limits)
    ax.plot(limits, limits, c=bpl.almost_black, lw=1, zorder=0)
    ax.equal_scale()
    ax.legend(loc=4)

    ax.add_labels(labels[x_suffix], labels[y_suffix])

fig.savefig(plot_name)
