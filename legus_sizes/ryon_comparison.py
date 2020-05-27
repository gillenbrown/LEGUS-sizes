import sys
from pathlib import Path

from astropy import table
import numpy as np
import betterplotlib as bpl

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
        catalogs[galaxy_name] = table.Table.read(item, format="ascii.ecsv")

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
catalogs["ngc1313-e"]["ID"] = [f"{i}-e" for i in catalogs["ngc1313-e"]["ID"]]
catalogs["ngc1313-w"]["ID"] = [f"{i}-w" for i in catalogs["ngc1313-w"]["ID"]]
catalogs["ngc628-e"]["ID"] = [f"{i}-e" for i in catalogs["ngc628-e"]["ID"]]
catalogs["ngc628-c"]["ID"] = [f"{i}-c" for i in catalogs["ngc628-c"]["ID"]]

# Then match the tables together. Using the inner join type is the strict intersection
# where the matched keys must match exactly
m_1313_e = table.join(catalogs["ngc1313-e"], ryon_1313, join_type="inner", keys="ID")
m_1313_w = table.join(catalogs["ngc1313-w"], ryon_1313, join_type="inner", keys="ID")
m_628_e = table.join(catalogs["ngc628-e"], ryon_628, join_type="inner", keys="ID")
m_628_c = table.join(catalogs["ngc628-c"], ryon_628, join_type="inner", keys="ID")

# ======================================================================================
#
# Making plots
#
# ======================================================================================
limits = 0.1, 100
# First we'll make a straight comparison
fig, axs = bpl.subplots(figsize=[14, 7], ncols=2)
for ax, method in zip(axs, ["Galfit", "CI"]):
    ax.errorbar(
        m_1313_e[f"r_eff_{method}"],
        m_1313_e["effective_radius_pc"],
        xerr=[m_1313_e[f"e_r_eff-_{method}"], m_1313_e[f"e_r_eff+_{method}"]],
        label="NGC1313-e",
    )
    ax.errorbar(
        m_1313_w[f"r_eff_{method}"],
        m_1313_w["effective_radius_pc"],
        xerr=[m_1313_w[f"e_r_eff-_{method}"], m_1313_w[f"e_r_eff+_{method}"]],
        label="NGC1313-w",
    )
    ax.errorbar(
        m_628_e[f"r_eff_{method}"],
        m_628_e["effective_radius_pc"],
        xerr=[m_628_e[f"e_r_eff-_{method}"], m_628_e[f"e_r_eff+_{method}"]],
        label="NGC628-e",
    )
    ax.errorbar(
        m_628_c[f"r_eff_{method}"],
        m_628_c["effective_radius_pc"],
        xerr=[m_628_c[f"e_r_eff-_{method}"], m_628_c[f"e_r_eff+_{method}"]],
        label="NGC628-c",
    )
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_limits(*limits, *limits)
    ax.plot(limits, limits, c=bpl.almost_black, lw=1, zorder=0)
    ax.equal_scale()
    ax.legend(loc=4)
    ax.add_labels(
        "Cluster $R_{eff}$ [pc] - Ryon+ 2017", "Cluster $R_{eff}$ [pc] - Me", method
    )
fig.savefig("comparison_plot.png")

# Then a ratio comparison
fig, axs = bpl.subplots(figsize=[14, 7], ncols=2)
for ax, method in zip(axs, ["Galfit", "CI"]):
    ax.errorbar(
        m_1313_e[f"r_eff_{method}"],
        m_1313_e["effective_radius_pc"] / m_1313_e[f"r_eff_{method}"],
        xerr=[m_1313_e[f"e_r_eff-_{method}"], m_1313_e[f"e_r_eff+_{method}"]],
        label="NGC1313-e",
    )
    ax.errorbar(
        m_1313_w[f"r_eff_{method}"],
        m_1313_w["effective_radius_pc"] / m_1313_w[f"r_eff_{method}"],
        xerr=[m_1313_w[f"e_r_eff-_{method}"], m_1313_w[f"e_r_eff+_{method}"]],
        label="NGC1313-w",
    )
    ax.errorbar(
        m_628_e[f"r_eff_{method}"],
        m_628_e["effective_radius_pc"] / m_628_e[f"r_eff_{method}"],
        xerr=[m_628_e[f"e_r_eff-_{method}"], m_628_e[f"e_r_eff+_{method}"]],
        label="NGC628-e",
    )
    ax.errorbar(
        m_628_c[f"r_eff_{method}"],
        m_628_c["effective_radius_pc"] / m_628_c[f"r_eff_{method}"],
        xerr=[m_628_c[f"e_r_eff-_{method}"], m_628_c[f"e_r_eff+_{method}"]],
        label="NGC628-c",
    )
    ax.set_yscale("log")
    ax.set_xscale("log")

    limits = 0.1, 30
    ax.set_limits(*limits, 0.1, 10)
    ax.plot(limits, [1, 1], c=bpl.almost_black, lw=1, zorder=0)
    ax.legend()
    ax.add_labels(
        "Cluster $R_{eff}$ [pc] - Ryon+ 2017", "$R_{eff, Me} / R_{eff, Ryon}$", method
    )

fig.savefig("comparison_ratio_plot.png")
