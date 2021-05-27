"""
generate_readme.py

Generate the readme file. I only do this programatically to make the formatting easier
on me and to make sure I document all columns in the public catalog.

Takes the following command line arguments:
- location to save the readme
- the public catalog.
"""
import sys
from pathlib import Path
from astropy import table

# Get the input arguments
output = Path(sys.argv[1])
catalog = table.Table.read(sys.argv[2], format="ascii.ecsv")

# set up the column names to make sure I include all of them, and that I do so in order
unused_colnames = catalog.colnames.copy()

# ======================================================================================
#
# write the introductory information to the header
#
# ======================================================================================
out_file = open(output, "w")

out_file.write(
    "# LEGUS-sizes\n"
    "This repository holds the code and the catalog used in Brown & Gnedin 2021. "
    "The catalog is the `cluster_sizes_brown_gnedin_21.txt` file here. I'll first "
    "describe the catalog, then describe the code used to generate the catalog.\n"
    "\n"
    "## Cluster Catalog\n"
    "The cluster catalog contains quantities calculated by LEGUS such as masses and "
    "radii ("
    "[Calzetti et al. 2015]"
    "(https://ui.adsabs.harvard.edu/abs/2015AJ....149...51C/abstract), "
    "[Adamo et al. 2017]"
    "(https://ui.adsabs.harvard.edu/abs/2017ApJ...841..131A/abstract), "
    "[Cook et al. 2019]"
    "(https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.4897C/abstract)"
    ") as well as the radii and other derived quantities from this paper. "
    "If using quantities from those papers, please cite them appropriately.\n"
    "\n"
    "I will not give a detailed description of the columns that come from LEGUS. "
    "Reference the papers above or the "
    "[public LEGUS catalogs]"
    "(https://archive.stsci.edu/prepds/legus/dataproducts-public.html) "
    "for more on that data.\n"
    "\n"
    "I do include all properties needed to duplicate the analysis done in the paper."
    "\n"
)

# ======================================================================================
#
# set up the items to group together
#
# ======================================================================================
groups = {
    "field": ["field"],
    "id": ["ID"],
    "galaxy": ["galaxy"],
    "distance": ["galaxy_distance_mpc", "galaxy_distance_mpc_err"],
    "galaxy_props": ["galaxy_stellar_mass", "galaxy_sfr", "galaxy_ssfr"],
    "pixel_scale": ["pixel_scale"],
    "ra_dec": ["RA", "Dec"],
    "xy_legus": ["x_pix_single", "y_pix_single"],
    "photometry": [
        "mag_F275W",
        "photoerr_F275W",
        "mag_F336W",
        "photoerr_F336W",
        "mag_F814W",
        "photoerr_F814W",
    ],
    "ci": ["CI"],
}

# ======================================================================================
#
# set up the descriptions for each group
#
# ======================================================================================
descriptions = {
    "field": "The identifier for the LEGUS field. Note that some galaxies are split "
    "over multiple fields (NGC 1313, NGC 4395, NGC 628, and NGC 7793), and that one "
    "field contains multiple galaxies (NGC 5194 and NGC 5195).",
    "id": "The cluster ID assigned by LEGUS. This was done on a field-by-field basis.",
    "galaxy": "The galaxy the cluster belongs to. NGC 5194 and NGC 5195 are separated"
    "manually (see Figure 1 of the paper). ",
    "distance": "Distance to the galaxy and its error. We use the TRGB distances to "
    "all LEGUS galaxies provided by "
    "[Sabbi et al. 2018]"
    "(https://ui.adsabs.harvard.edu/abs/2018ApJS..235...23S/abstract)"
    ", except for NGC 1566. See the end of Section 2.4 for more on distances used.",
    "galaxy_props": "Stellar masses, star formation rates, and  specific star "
    "formation rates of the host galaxy, from "
    "[Calzetti et al. 2015]"
    "(https://ui.adsabs.harvard.edu/abs/2015AJ....149...51C/abstract)."
    "SFR is obtained from *GALEX* far-UV corrected for dust attenuation, as described "
    "in "
    "[Lee et al. 2009](https://ui.adsabs.harvard.edu/abs/2009ApJ...706..599L/abstract)"
    ", and stellar mass from extinction-corrected B-band luminosity and color "
    "information, as described in "
    "[Bothwell et al. 2009]"
    "(https://ui.adsabs.harvard.edu/abs/2009MNRAS.400..154B/abstract)"
    "and using the mass-to-light ratio models of "
    "[Bell & de Jong 2001]"
    "(https://ui.adsabs.harvard.edu/abs/2001ApJ...550..212B/abstract).",
    "pixel_scale": "Pixel scale for the image. All are nearly 39.62 mas/pixel.",
    "ra_dec": "Right ascension and declination from the LEGUS catalog.",
    "xy_legus": "X/Y pixel position of the cluster from the LEGUS catalog.",
    "photometry": "LEGUS photometry.",
    "ci": "Concentration Index (CI), defined as the which is the magnitude difference "
    "between apertures of radius 1 pixel and 3 pixels. A cut was used to separate "
    "stars from clusters.",
}

# ======================================================================================
#
# write these to the catalog!
#
# ======================================================================================
for group_name, group in groups.items():
    # validate that we got the colnames in order as we write them
    for col in group:
        assert col == unused_colnames[0]
        del unused_colnames[0]

    # then go through and write the column names to the file.
    out_file.write("**")
    for idx_c in range(len(group)):
        out_file.write(f"`{group[idx_c]}`")
        # if we are not at the last column name of the group, we need to write a comma
        if idx_c != len(group) - 1:
            out_file.write(", ")
        # if we are the last column name, put the newline
        else:
            out_file.write("**\n")

    # then write the description
    out_file.write(descriptions[group_name])
    out_file.write("\n\n")

# check that we got everything
try:
    assert len(unused_colnames) == 0
except AssertionError:
    print(unused_colnames)
    raise RuntimeError
out_file.close()
