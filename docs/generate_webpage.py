"""
generate_webpage.py

Generate the readme file. I only do this programatically to make the formatting easier
on me and to make sure I document all columns in the public catalog.

Takes the following command line arguments:
- location to save the readme
- the public catalog.
"""
import sys
import re
from pathlib import Path
from astropy import table

# Get the input arguments
out_file = open(sys.argv[1], "w")
catalog = table.Table.read(sys.argv[2], format="ascii.ecsv")

# set up the column names to make sure I include all of them, and that I do so in order
unused_colnames = catalog.colnames.copy()

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
    "eta": ["power_law_slope", "power_law_slope_e-", "power_law_slope_e+"],
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
    "(https://ui.adsabs.harvard.edu/abs/2015AJ....149...51C/abstract). "
    "SFR is obtained from *GALEX* far-UV corrected for dust attenuation, as described "
    "in "
    "[Lee et al. 2009](https://ui.adsabs.harvard.edu/abs/2009ApJ...706..599L/abstract)"
    ", and stellar mass from extinction-corrected B-band luminosity and color "
    "information, as described in "
    "[Bothwell et al. 2009]"
    "(https://ui.adsabs.harvard.edu/abs/2009MNRAS.400..154B/abstract) "
    "and using the mass-to-light ratio models of "
    "[Bell & de Jong 2001]"
    "(https://ui.adsabs.harvard.edu/abs/2001ApJ...550..212B/abstract). ",
    "pixel_scale": "Pixel scale for the image. All are nearly 39.62 mas/pixel.",
    "ra_dec": "Right ascension and declination from the LEGUS catalog.",
    "xy_legus": "X/Y pixel position of the cluster from the LEGUS catalog.",
    "ci": "Concentration Index (CI), defined as the which is the magnitude difference "
    "between apertures of radius 1 pixel and 3 pixels. A cut was used to separate "
    "stars from clusters.",
}

# ======================================================================================
#
# write these to the catalog!
#
# ======================================================================================
# What I'll do is have placeholders for specific column. I'll identify those
# placeholders and replace them with the actual column name
template_flag = "__template__"
template_regex = re.compile(f"{template_flag}\w*{template_flag}")

with open(sys.argv[3], "r") as template:
    for line in template:
        # see if this is a line where a template should be inserted
        if template_regex.match(line.strip()):
            # grab the name of the group being indicated here
            group_name = line.strip().replace(template_flag, "")
            # then get the columns that belong to this group
            group = groups[group_name]

            # validate that we got the colnames in order as we write them
            for col in group:
                try:
                    assert col == unused_colnames[0]
                except AssertionError:
                    raise RuntimeError(f"{col} bad order, should be:", unused_colnames)
                # if successful, delete the column
                del unused_colnames[0]

            # then go through and write the column names to the file.
            out_file.write("**")
            for idx_c in range(len(group)):
                out_file.write(f"`{group[idx_c]}`")
                # if we are not at the last column name of the group, we need to
                # write a comma
                if idx_c != len(group) - 1:
                    out_file.write(", ")
                # if we are the last column name, put the newline
                else:
                    out_file.write("**\n\n")

            # then write the description
            out_file.write(descriptions[group_name])
            out_file.write("\n\n")
        # if it does not match the template, just write the line
        else:
            out_file.write(line)

# check that we got everything
try:
    assert len(unused_colnames) == 0
except AssertionError:
    raise RuntimeError("These coluns are unused:", unused_colnames)

out_file.close()
